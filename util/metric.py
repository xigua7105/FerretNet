

def topk_accuracy(output, target, top_k=(1, 5)):
    max_k = max(top_k)
    _, pred = output.topk(max_k, 1, True, True)
    correct = pred.eq(target.view(target.size(0), -1).expand_as(pred))
    return [correct[:, :k].sum().item() for k in top_k]
