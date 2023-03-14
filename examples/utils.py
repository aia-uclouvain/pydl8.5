import graphviz


class PlotGraphviz:

    def __init__(self, dot_string):
        self.dot_string = dot_string

    def _repr_html_(self):
        return graphviz.Source(self.dot_string)._repr_svg_()