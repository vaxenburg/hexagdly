import torch

def conv2d(input, weight, bias=None, stride=1):
    """Functional 2D hexagonal convolution.

    The hexagonal convolution operation is reduced to a number of
    rectangular convolution operations performed by torch.nn.functional.conv2d

    This function is a wrapper for:
        operation_with_single_hexbase_stride
        operation_with_arbitrary_stride

    The implementation is based on the Conv2d class from
    the hexagdly package, https://github.com/ai4iacts/hexagdly

    For more information and input/output format description, please see:
    https://www.sciencedirect.com/science/article/pii/S2352711018302723
    https://github.com/ai4iacts/hexagdly

    Args:
        input (tensor): Input tensor of shape (minibatch, in_channels, H_in, W_in).
        weight (list of tensors): List of columns of the hexagonal kernel(s),
            [..., kernel-2, kernel-1, kernel0, kernel1, kernel2, ...]
            Each element in the list is one kernel column of shape:
                (out_channels, in_channels, column_length)
            For example:
                If kernel size is 1, column_length is 2, 3, 2.
                If kernel size is 2, column_length is 3, 4, 5, 4, 3.
        bias (tensor): Optional bias tensor of shape (out_channels), default=None.
        stride (int): Optional stride of the convolving kernel, default=1.

    Returns:
        Tensor of shape (minibatch, out_channels, H_out, W_out).
    """

    in_channels = input.shape[1]
    out_channels = weight[0].shape[0]
    hexbase_size = weight[0].shape[2] - 1 # this is the kernel size
    hexbase_stride = stride
    process = torch.nn.functional.conv2d
    combine = torch.add

    # Prepare kernel's columns
    kernel = []
    for i in range(hexbase_size + 1):
        if i==0:
            kernel.append(weight[hexbase_size][:, :, :, None]) # kernel0
        else:
            kernel.append(
                torch.stack((weight[hexbase_size - i],
                             weight[hexbase_size + i]), axis=3)
            ) # kernel1, kernel2, etc..

    if hexbase_stride == 1:
        operation = operation_with_single_hexbase_stride
    else:
        operation = operation_with_arbitrary_stride

    return operation(input, kernel, hexbase_stride,
                     bias, process, combine,
                     dimensions=2, depth_stride=None)


def operation_with_arbitrary_stride(input, kernel, hexbase_stride,
                                    bias, process, combine,
                                    dimensions=2, depth_stride=None):
    """General implementation of an operation with a hexagonal kernel"""

    assert (
        input.size(-2) - (hexbase_stride // 2) >= 0
    ), "Too few rows to apply hex conv with the stride that is set"
    odd_columns = None
    even_columns = None

    hexbase_size = len(kernel) - 1

    odd_columns_slices = []
    even_columns_slices = []
    even_columns_pads = []
    odd_columns_pads = []

    input_size_is_known = False

    for i in range(hexbase_size + 1):
        dilation_base = (1, 1) if i == 0 else (1, 2 * i)

        if not input_size_is_known:
            slices, pads = _shape_for_odd_columns(input.size(), i, hexbase_stride, hexbase_size)
            odd_columns_slices.append(slices)
            odd_columns_pads.append(pads)
            slices, pads = _shape_for_even_columns(input.size(), i, hexbase_stride, hexbase_size)
            even_columns_slices.append(slices)
            even_columns_pads.append(pads)
            if i == hexbase_size:
                input_size_is_known = True

        if odd_columns is None:
            odd_columns = process(
                _get_padded_input(
                    _get_sliced_input(input, odd_columns_slices[i], dimensions),
                    odd_columns_pads[i],
                    dimensions,
                ),
                kernel[i],
                dilation=_get_dilation(dilation_base, dimensions),
                stride=_get_stride(dimensions,
                                   hexbase_stride,
                                   depth_stride),
                bias=bias
            )
        else:
            odd_columns = combine(
                odd_columns,
                process(
                    _get_padded_input(
                        _get_sliced_input(input, odd_columns_slices[i], dimensions),
                        odd_columns_pads[i],
                        dimensions,
                    ),
                    kernel[i],
                    dilation=_get_dilation(dilation_base, dimensions),
                    stride=_get_stride(dimensions,
                                       hexbase_stride,
                                       depth_stride),
                ),
            )

        if even_columns is None:
            even_columns = process(
                _get_padded_input(
                    _get_sliced_input(input, even_columns_slices[i], dimensions),
                    even_columns_pads[i],
                    dimensions,
                ),
                kernel[i],
                dilation=_get_dilation(dilation_base, dimensions),
                stride=_get_stride(dimensions,
                                   hexbase_stride,
                                   depth_stride),
                bias=bias
            )
        else:
            even_columns = combine(
                even_columns,
                process(
                    _get_padded_input(
                        _get_sliced_input(input, even_columns_slices[i], dimensions),
                        even_columns_pads[i],
                        dimensions,
                    ),
                    kernel[i],
                    dilation=_get_dilation(dilation_base, dimensions),
                    stride=_get_stride(dimensions,
                                       hexbase_stride,
                                       depth_stride),
                ),
            )

    concatenated_columns = torch.cat(
        (odd_columns, even_columns), 1 + dimensions
    )

    n_odd_columns = odd_columns.size(-1)
    n_even_columns = even_columns.size(-1)
    if n_odd_columns == n_even_columns:
        order = [
            int(i + x * n_even_columns)
            for i in range(n_even_columns)
            for x in range(2)
        ]
    else:
        order = [
            int(i + x * n_odd_columns)
            for i in range(n_even_columns)
            for x in range(2)
        ]
        order.append(n_even_columns)

    return _get_ordered_output(concatenated_columns, order, dimensions)


def operation_with_single_hexbase_stride(input, kernel, hexbase_stride,
                                         bias, process, combine,
                                         dimensions=2, depth_stride=None):
    """A slightly faster, case specific implementation of the hexagonal convolution"""

    hexbase_size = len(kernel) - 1
    columns_mod2 = input.size(-1) % 2
    odd_kernels_odd_columns = []
    odd_kernels_even_columns = []
    even_kernels_all_columns = []

    even_kernels_all_columns = process(
        _get_padded_input(input, [0, 0, hexbase_size, hexbase_size], dimensions),
        kernel[0],
        stride=(1, 1) if dimensions == 2 else (depth_stride, 1, 1),
        bias=bias
    )
    if hexbase_size >= 1:
        odd_kernels_odd_columns = process(
            _get_padded_input(
                input, [1, columns_mod2, hexbase_size, hexbase_size - 1], dimensions
            ),
            kernel[1],
            dilation=_get_dilation((1, 2), dimensions),
            stride=_get_stride(dimensions,
                               hexbase_stride,
                               depth_stride),
        )
        odd_kernels_even_columns = process(
            _get_padded_input(
                input,
                [0, 1 - columns_mod2, hexbase_size - 1, hexbase_size],
                dimensions,
            ),
            kernel[1],
            dilation=_get_dilation((1, 2), dimensions),
            stride=_get_stride(dimensions,
                               hexbase_stride,
                               depth_stride),
        )

    if hexbase_size > 1:
        for i in range(2, hexbase_size + 1):
            if i % 2 == 0:
                even_kernels_all_columns = combine(
                    even_kernels_all_columns,
                    process(
                        _get_padded_input(
                            input,
                            [
                                i,
                                i,
                                hexbase_size - int(i / 2),
                                hexbase_size - int(i / 2),
                            ],
                            dimensions,
                        ),
                        kernel[i],
                        dilation=_get_dilation((1, 2 * i), dimensions),
                        stride=(1, 1)
                        if dimensions == 2
                        else (depth_stride, 1, 1),
                    ),
                )
            else:
                x = hexbase_size + int((1 - i) / 2)
                odd_kernels_odd_columns = combine(
                    odd_kernels_odd_columns,
                    process(
                        _get_padded_input(
                            input, [i, i - 1 + columns_mod2, x, x - 1], dimensions,
                        ),
                        kernel[i],
                        dilation=_get_dilation((1, 2 * i), dimensions),
                        stride=_get_stride(dimensions,
                                           hexbase_stride,
                                           depth_stride),
                    ),
                )
                odd_kernels_even_columns = combine(
                    odd_kernels_even_columns,
                    process(
                        _get_padded_input(
                            input, [i - 1, i - columns_mod2, x - 1, x], dimensions,
                        ),
                        kernel[i],
                        dilation=_get_dilation((1, 2 * i), dimensions),
                        stride=_get_stride(dimensions,
                                           hexbase_stride,
                                           depth_stride),
                    ),
                )

    odd_kernels_concatenated_columns = torch.cat(
        (odd_kernels_odd_columns, odd_kernels_even_columns), 1 + dimensions
    )

    n_odd_columns = odd_kernels_odd_columns.size(-1)
    n_even_columns = odd_kernels_even_columns.size(-1)
    if n_odd_columns == n_even_columns:
        order = [
            int(i + x * n_even_columns)
            for i in range(n_even_columns)
            for x in range(2)
        ]
    else:
        order = [
            int(i + x * n_odd_columns)
            for i in range(n_even_columns)
            for x in range(2)
        ]
        order.append(n_even_columns)

    return combine(
        even_kernels_all_columns,
        _get_ordered_output(odd_kernels_concatenated_columns, order, dimensions),
    )


def _shape_for_odd_columns(input_size, kernel_number, hexbase_stride, hexbase_size):
    slices = [None, None, None, None]
    pads = [0, 0, 0, 0]
    # left
    pads[0] = kernel_number
    # right
    pads[1] = max(
        0, kernel_number - ((input_size[-1] - 1) % (2 * hexbase_stride))
    )
    # top
    pads[2] = hexbase_size - int(kernel_number / 2)
    # bottom
    constraint = (
        input_size[-2]
        - 1
        - int(
            (input_size[-2] - 1 - int(hexbase_stride / 2))
            / hexbase_stride
        )
        * hexbase_stride
    )
    bottom = (hexbase_size - int((kernel_number + 1) / 2)) - constraint
    if bottom >= 0:
        pads[3] = bottom
    else:
        slices[1] = bottom

    return slices, pads


def _shape_for_even_columns(input_size, kernel_number, hexbase_stride, hexbase_size):
    slices = [None, None, None, None]
    pads = [0, 0, 0, 0]
    # left
    left = kernel_number - hexbase_stride
    if left >= 0:
        pads[0] = left
    else:
        slices[2] = -left
    # right
    pads[1] = max(
        0,
        kernel_number
        - ((input_size[-1] - 1 - hexbase_stride) % (2 * hexbase_stride)),
    )
    # top
    top_shift = -(kernel_number % 2) if (hexbase_stride % 2) == 1 else 0
    top = (
        (hexbase_size - int(kernel_number / 2))
        + top_shift
        - int(hexbase_stride / 2)
    )
    if top >= 0:
        pads[2] = top
    else:
        slices[0] = -top
    # bottom
    bottom_shift = 0 if (hexbase_stride % 2) == 1 else -(kernel_number % 2)
    pads[3] = max(
        0,
        hexbase_size
        - int(kernel_number / 2)
        + bottom_shift
        - (
            (input_size[-2] - int(hexbase_stride / 2) - 1)
            % hexbase_stride
        ),
    )

    return slices, pads


def _get_padded_input(input, pads, dimensions):
    if dimensions == 2:
        return torch.nn.ZeroPad2d(tuple(pads))(input)
    elif dimensions == 3:
        return torch.nn.ConstantPad3d(tuple(pads + [0, 0]), 0)(input)


def _get_sliced_input(input, slices, dimensions):
    if dimensions == 2:
        return input[:, :, slices[0] : slices[1], slices[2] : slices[3]]
    elif dimensions == 3:
        return input[:, :, :, slices[0] : slices[1], slices[2] : slices[3]]


def _get_dilation(dilation_2d, dimensions):
    if dimensions == 2:
        return dilation_2d
    elif dimensions == 3:
        return tuple([1] + list(dilation_2d))


def _get_stride(dimensions, hexbase_stride, depth_stride):
    if dimensions == 2:
        return (hexbase_stride, 2 * hexbase_stride)
    elif dimensions == 3:
        return (depth_stride, hexbase_stride, 2 * hexbase_stride)


def _get_ordered_output(input, order, dimensions):
    if dimensions == 2:
        return input[:, :, :, order]
    elif dimensions == 3:
        return input[:, :, :, :, order]
