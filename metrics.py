def accuracy(predicted_scores, y):
    predicted_classes = predicted_scores.argmax(dim=1, keepdim=True)
    correct = (predicted_classes == y.view_as(predicted_classes)).sum()
    return correct.float() / y.shape[0]