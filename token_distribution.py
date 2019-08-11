from nltk import FreqDist
from nltk.corpus import stopwords
import plotly.graph_objs as go


stopwords_default = stopwords.words('english')

layout = dict(
    title=go.layout.Title(
        text="Token Frequency",
        xref="paper"
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Tokens",
            font=dict(
                family="Courier New, monospace",
                size=20,
                color="Black"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Frequency",
            font=dict(
                family="Courier New, monospace",
                size=20,
                color="Black"
            )
        )
    )
)


class TokenDist(object):
    def __init__(self,
                 texts,
                 rev=True,
                 tokenizer=None,
                 stops=stopwords_default):
        self._texts = texts
        self._rev = rev
        self._freq_dict = None
        self._tokenizer = tokenizer
        self._stops = stops

    def get_freq_dict(self):
        if self._freq_dict is None:
            fd = FreqDist([tok for sent in self._texts for tok in self._tokenizer.gap_split(sent, return_spans=False)
                           if tok.lower() not in self._stops])
            fd = dict(sorted([(k, v) for k, v in fd.items()], reverse=self._rev, key=lambda x: x[1]))
            self._freq_dict = fd
            return fd
        else:
            return self._freq_dict

    def return_go_hist(self, top_n = 30, start = 0):
        fig = go.Figure()
        low_ = start
        high_ = start + top_n
        fig.add_trace(go.Histogram(histfunc="sum",
                                   y=list(self.get_freq_dict().values())[low_:high_],
                                   x=list(self.get_freq_dict().keys())[low_:high_], name="count"))
        fig.update_xaxes(tickangle=40,
                         tickfont=dict(family='Rockwell', color='crimson', size=14))

        fig.layout.update(layout)
        return fig
