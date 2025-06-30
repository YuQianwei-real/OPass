from ..expr import *
from ..spec import TypeSpec, Op, rank_ran, dim_ran


def create_identity():
    return TypeSpec(
        attrs=[],
        in_num=1,
        in_ranks=[Var()],
        in_dtypes=[Var()],
        in_shapes=[List(IN[0].rank, lambda _: Var(tmpl=True))],
        extra=[],
        out_num=1,
        out_ranks=[IN[0].rank],
        out_dtypes=[IN[0].dtype],
        out_shapes=[IN[0].shape]
    )


Op('negative', create_identity)
Op('abs', create_identity)
Op('ceil', create_identity)
Op('floor', create_identity)
Op('round', create_identity)
Op('trunc', create_identity)
Op('exp', create_identity)
Op('sin', create_identity)
Op('cos', create_identity)
Op('tan', create_identity)
Op('sigmoid', create_identity)
Op('tanh', create_identity)

# Add by nie
Op('variance', None)
Op('fast_exp', None)
Op('sqrt', None)
Op('fast_tanh', None)
Op('nn.fast_softmax', None)
Op('rsqrt', None)
Op('annotation.stop_fusion', None)
Op('layout_transform', None)
Op('zeros_like', None)
Op('zeros', None)
Op('ones', None)
Op('cast', None)
Op('collapse_sum_like', None)
Op('reshape_like', None)
Op('collapse_sum_to', None)
Op('power', None)
Op('broadcast_to_like', None)
Op('ones_like', None)
Op('cast_like', None)
Op('broadcast_to', None)
Op('nn.batch_matmul', None)
Op('fast_erf', None)
Op('erf', None)
Op('clip', None)
Op('stack', None)
Op('nn.dropout', None)
Op('nn.contrib_dense_pack', None)
Op('full', None)
Op('full_like', None)
Op('nn.avg_pool2d_grad', None)
Op('less', None)
Op('where', None)
Op('tile', None)
Op('nn.max_pool2d_grad', None)
Op('on_device', None)
Op('nn.global_max_pool2d', None)
Op('shape_of', None)
Op('dyn.tile', None)
Op('qnn.dequantize', None)
Op('image.resize2d', None)
Op('qnn.quantize', None)
Op('qnn.dense', None)
Op('qnn.subtract', None)
Op('qnn.batch_matmul', None)
Op('nn.space_to_batch_nd', None)
Op('qnn.concatenate', None)
Op('nn.batch_to_space_nd', None)
Op('take', None)
Op('dyn.ones', None)
Op('dyn.nn.upsampling3d', None)
Op('dyn.nn.upsampling', None)
Op('argsort', None)
Op('argmin', None)
Op('dyn.squeeze', None)
Op('dyn.broadcast_to', None)
Op('nn.depth_to_space', None)
Op('qnn.conv2d', None)
Op('argmax', None)
Op('dyn.image.resize2d', None)
Op('one_hot', None)
Op('left_shift', None)
Op('qnn.add', None)
Op('dyn.reshape', None)
Op('qnn.requantize', None)
Op('topk', None)
Op('qnn.mul', None)
Op('dyn.sparse_to_dense', None)
Op('dyn.nn.pad', None)
Op('ndarray_size', None)
Op('gather_nd', None)
Op('contrib_reverse_reshape', None)
Op('dyn.full', None)
Op('nn.conv2d_backward_weight', None)

# Virtual operators
Op('let', None)
Op('func', None)
Op('def', None)
Op('call', None)
Op('tuple', None)
Op('getitem', None)

Op('ref', None)
Op('ref_write', None)
Op('ref_read', None)

# Op('UnknownOp', create_identity)


def _create_bcast():
    m = IN[0].rank
    n = IN[1].rank
    if TypeSpec.for_graph:
        return TypeSpec(
            attrs=[],
            in_num=2,
            in_ranks=[Var(), Var(ran=iran(2, m))],
            in_dtypes=List(2, lambda _: Var()),
            in_shapes=[
                List(m, lambda _: Var(ran=dim_ran, tmpl=True)),
                List(n, lambda _: Var(ran=dim_ran, tmpl=True))
            ],
            extra=[
                ForAll(Range(end=n), lambda i: Or(
                    IN[0].shape[m - i - 1] == IN[1].shape[n - i - 1],
                    IN[0].shape[m - i - 1] == 1,
                    IN[1].shape[n - i - 1] == 1,
                ))
            ],
            out_num=1,
            out_ranks=[m],
            out_dtypes=[IN[0].dtype],
            out_shapes=[Concat(
                IN[0].shape[Range(end=m - n)],
                List(n, lambda i: IN[0].shape[m - n + i].max(IN[1].shape[i]))
            )],
        )
    return TypeSpec(
        attrs=[],
        in_num=2,
        in_ranks=List(2, lambda _: Var(ran=rank_ran, tmpl=True)),
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            List(m, lambda _: Var(ran=dim_ran, tmpl=True)),
            List(n, lambda _: Var(ran=dim_ran, tmpl=True))
        ],
        extra=[
            ForAll(Range(end=m.min(n)), lambda i: Or(
                IN[0].shape[m - i - 1] == IN[1].shape[n - i - 1],
                IN[0].shape[m - i - 1] == 1,
                IN[1].shape[n - i - 1] == 1
            ))
        ],
        out_num=1,
        out_ranks=[m.max(n)],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            Cond(
                m >= n,
                Concat(
                    IN[0].shape[Range(end=m - n)],
                    List(n, lambda i: IN[0].shape[m - n + i].max(IN[1].shape[i]))
                ),
                Concat(
                    IN[1].shape[Range(end=n - m)],
                    List(m, lambda i: IN[0].shape[i].max(IN[1].shape[n - m + i]))
                )
            )
        ]
    )


Op('add', _create_bcast)
Op('subtract', _create_bcast)
Op('multiply', _create_bcast)
Op('divide', _create_bcast)
Op('maximum', _create_bcast)
Op('minimum', _create_bcast)
