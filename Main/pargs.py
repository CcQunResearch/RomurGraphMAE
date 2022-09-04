import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Weibo')
    parser.add_argument('--vector_size', type=int, help='word embedding size', default=200)
    parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabel data train size', default=20000)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--ft_runs', type=int, default=5)

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--unsup_bs_ratio', type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=128, help="number of hidden units")
    parser.add_argument("--residual", type=str2bool, default=True, help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=0.2, help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0.1, help="attention dropout")
    # 'layernorm'或者'layernorm'影响不大
    parser.add_argument("--norm", type=str, default='layernorm')
    parser.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--concat_hidden", type=str2bool, default=True)
    parser.add_argument("--pooling", type=str, default="max")

    parser.add_argument("--encoder", type=str, default="gcn")
    parser.add_argument("--decoder", type=str, default="gcn")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ft_lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ft_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lamda', dest='lamda', type=float, default=0.001)

    parser.add_argument('--k', type=int, default=10000)

    args = parser.parse_args()
    return args
