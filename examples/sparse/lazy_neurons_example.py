import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
from split_learn.modifiers import TopKModifier, RandomTopKModifier


n_points = 20


xs, ys = np.meshgrid(np.arange(-n_points, n_points) * 1.3 / n_points,
                     np.arange(-n_points, n_points) * 1.3/n_points)

def lazy_neurons_example():
    x1, y1 = torch.tensor([1., 0.]), torch.tensor(1.)
    x2, y2 = torch.tensor([0.5, 1]), torch.tensor(-1.)



    records = {
        "w1": [],
        "x21w1": [],
        "x22w1": []
    }

    topk = TopKModifier(0.5, [2])
    # topk = RandomTopKModifier(0.5).modify_forward_train
    def loss_func(x, y):
        hidden = topk(torch.stack([w1 * x[0], w2 * x[1]]).view(1, -1))
        return F.mse_loss(torch.tanh(torch.sum(hidden)), y)

    grads_w1 = []
    grads_w2 = []

    # Compute gradients

    n_ws = 5

    w1s, w2s = np.meshgrid(np.arange(-n_ws, n_ws) * 1.2 / n_ws,
                         np.arange(-n_ws, n_ws) * 1.2 / n_ws)

    for i in range(w1s.shape[0]):
        for j in range(w1s.shape[1]):
            w1, w2 = torch.tensor(w1s[i, j], dtype=torch.float, requires_grad=True), \
                     torch.tensor(w2s[i, j], dtype=torch.float, requires_grad=True)
            loss = loss_func(x1, y1) + loss_func(x2, y2)
            target = - loss
            w1_grad = torch.autograd.grad(target, [w1], retain_graph=True)[0]
            w2_grad = torch.autograd.grad(target, [w2])[0].item()
            grads_w1.append(w1_grad)
            grads_w2.append(w2_grad)


    w1 = torch.tensor(1., requires_grad=True)
    w2 = torch.tensor(-0.1, requires_grad=True)


    optimizer = SGD([w1, w2], 0.01)

    for i in range(10):
        print(f"Current weights: {w1.item(), w2.item()}")
        records["w1"].append(w1.item())
        for j in range(10):
            optimizer.zero_grad()
            loss = loss_func(x1, y1) + loss_func(x2, y2)
            print(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

    return [w1s, w2s, grads_w1, grads_w2], records


def get_loss_surface():
    x1, y1 = np.array([1., 0.]), 1.
    x2, y2 = np.array([0.5, 1]), -1.

    def sample_loss(w1, w2, xs, y):

        h = (np.abs(w1 * xs[0]) > np.abs(w2 * xs[1])) * w1 * xs[0] + \
            (np.abs(w1 * xs[0]) <= np.abs(w2 * xs[1])) * w2 * xs[1]
        return np.square(np.tanh(h) - y)

    def total_loss(w1, w2):
        return (sample_loss(w1, w2, x1, y1) + sample_loss(w1, w2, x2, y2)) / 2

    return total_loss(xs, ys)




if __name__ == '__main__':
    (w1s, w2s, w1gs, w2gs), records = lazy_neurons_example()

    plt.figure(figsize=(5, 5))
    loss_surface = get_loss_surface()
    # bg = plt.gca().imshow(loss_surface, origin='lower')
    cset = plt.contourf(xs, ys, loss_surface, 40, cmap='gray_r')
    cset2 = plt.contour(xs, ys, loss_surface, 20, colors='k', linewidths=0.4, linestyles='dashed')

    plt.quiver(w1s, w2s, w1gs, w2gs)



    x_axis = np.arange(-n_points, n_points) * 1.3 / n_points
    w2_high = np.abs(x_axis * 0.5)
    w2_low = - np.abs(x_axis * 0.5)
    plt.fill_between(x_axis, w2_low, w2_high, step="pre", color='blue', alpha=0.4)


    plt.arrow(records["w1"][0], -0.1, records["w1"][-1] - records["w1"][0], 0,
              head_width=0.07, color='red', width=0.01)

    plt.xlabel('$w_1$', fontsize=15)
    plt.ylabel('$w_2$', fontsize=15)
    plt.savefig("topk-example-gradient-field.pdf", bbox_inches='tight')


    # map = plt.colorbar(bg, ax=plt.gca())
    # map.add_lines(contour)
    plt.show()
    # lazy_neurons_example()