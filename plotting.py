import matplotlib.pyplot as plt
from IPython import display

plt.ion()


class Plot:
    def clean(self) -> 'Plot':
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        return self

    def plot(self, x, y=None, *args, **kwargs) -> 'Plot':
        if y is None:
            plt.plot(x, *args, **kwargs)
        else:
            plt.plot(x, y, *args, **kwargs)
        return self

    def title(self, title: str, *args) -> 'Plot':
        plt.title(title, *args)
        return self

    def labels(self, xlabel: str, ylabel: str) -> 'Plot':
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return self

    def text(self, x: int, y: int, text: str) -> 'Plot':
        plt.text(x, y, text)
        return self

    def pause(self, time: float | int) -> 'Plot':
        plt.pause(time)
        return self
