### CODE STARTS
tables = [
    [('gpu',), ["1080ti"]],
    [('downstream_type',), ["modelnet40"]],
    [('ngpus',), ["1"]],
    [('perturb_types',), ["drop,scale,rotate"]],
    [('valid_shape_loss_lmbda',), [0.1, 1.0, 10.0, 100.0]],
    [('self_sup_lmbda',), [1.0]],
    [('perturb_amount',), [0.5]],
]
### CODE ENDS
