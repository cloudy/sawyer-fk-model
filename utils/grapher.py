import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'figure.max_open_warning': 0})

def plot_performance(hist, ptitle):
    print("Generating performance plot...")
    k = list(hist.keys())
    range_epochs = range(0, len(hist[k[0]]))
    fig = plt.figure()
    print(k)
    plt.title("%s, %s" % (ptitle.split('/')[-1], 
        ' '.join([k_+':'+str(round(hist[k_][-1], 3))+',' for k_ in k])),
        fontsize=12)
    plt.xlabel("epoch")
    plt.ylabel("loss/accuracy")
    plt.ylim(0, 1)
    for res in hist.keys():
        plt.plot(range_epochs, hist[res], label=res)
    plt.legend(loc='upper right')
    print("Performance plot generated. ")
    return fig

def save_plots(figs, filename="dataplot.pdf"):
    with PdfPages(filename) as pdf:
        for fig in figs:
            pdf.savefig(fig)
    print("Plots were saved as: %s" % filename)


