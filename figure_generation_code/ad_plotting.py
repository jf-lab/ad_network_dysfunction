
import sys
import numpy as np
sys.path.append("~/scope-recorder")
import warnings
from scipy.optimize import OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)

import dill as pickle
import seaborn as sns
from scipy.stats import pearsonr, gaussian_kde, ks_2samp, ttest_ind, norm, ttest_1samp


from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm
from scipy.stats import ttest_ind
from scipy.stats import gaussian_kde

from scipy.optimize import curve_fit, minimize

import statsmodels.formula.api as smf
import pandas as pd
#from patsy import dmatrices


import os
import base64
from xml.etree import ElementTree as et

def open_pickle(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)
    
    
pwd = os.path.dirname(os.path.realpath(__file__))


def map_res(f, res):
    return [[[f(a) for a in fa if a is not None and f(a) is not None] for fa in s] for s in res]

def zipwith_res(f, res1, res2):
    return [[[f(a1,a2) for a1, a2 in zip(fa1, fa2) if a1 is not None and a2 is not None] for fa1, fa2 in zip(s1, s2)] for s1, s2 in zip(res1, res2)]

def zipwith_n_res(f, *res):
    return [[[f(*a) for a in zip(*fa)] for fa in zip(*s)] for s in zip(*res)]


sns.set_style("white")
fontsize= 12
symbolfontsize = fontsize + 4
labelfontsize = fontsize + 3
inset_label_size = fontsize - 2
matplotlib.rcParams.update({"font.size":fontsize,
                            "legend.fontsize":fontsize,
                            "axes.titlesize":fontsize,
                            "axes.labelsize":fontsize,
                            "xtick.labelsize":fontsize,
                            "ytick.labelsize":fontsize,
                            "xtick.direction":"out",
                            "ytick.direction":"out",
                           })
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['mathtext.fontset'] = 'stixsans'
matplotlib.rcParams.update({"axes.spines.right": False,
                            "axes.spines.top": False,
                            "xtick.major.size": 4,
                            "ytick.major.size": 4,
                            "xtick.major.pad" : 1,
                            "ytick.major.pad" : 1,
                            "figure.frameon": False,
                            "hatch.linewidth": 3,
                           })
glua2 = "#ffa040"
veh = "#a0a0a4"
gray = "#606060"
#glua2 = "#fb6a0a"
black = (0,0,0,1)
transparent = (1,1,1,0)
    
FULL_FIGURE = True 
DOWN_SAMPLING = False
EXPORT_RES = True
LATEX = False  
PRISM = True
if LATEX:
    colorset = [  "#336da6","#2bab7e","#ff7e40", "#ffb140"] #WT, WT-Glu, Tg, Tg-Glu
    line_colors = colorset
    face_colors = colorset
    line_styles = ["-","-", "-", "-"]
if PRISM:
    colorset = [veh, glua2, "#d4d4d8", "#fccb99"]
    line_colors = [veh, glua2, veh, glua2]
    line_styles = ["-", "-", "-", "-"]
    face_colors = [veh, glua2, transparent, transparent]
    matplotlib.rcParams.update({
                                "font.sans-serif": ["Arial"],
                                "font.family": "sans-serif",
                               })
    
line_width = 1
axes_line_width = 1
line_width_bar = line_width * 2
line_width_dist = line_width * 2
fig_wspace = 0.2
trace_colors = colorset
stat_bar_color = veh
hatchset = [None, '//', 'xx', '-', '/', '-', '*', 'x', '\\', 'o', 'O', '.']
session_hatchset = [None, None, '//', None, None, '//']
hatch_colors = ['#57575a','#cc7f32',veh, glua2]

matplotlib.rcParams.update({"axes.linewidth": axes_line_width,
                            "ytick.major.width": axes_line_width,
                            "xtick.major.width": axes_line_width,
                            })

fig_path = "/tmp/fig.svg"
fig2_path = "/tmp/fig2.svg"
supp_fig_path = "/tmp/supp_fig.svg"
supp_fig2_path = "/tmp/supp_fig2.svg"
misc_dir = os.path.join(pwd, "..", "misc")
# In[3]:
lbw = 2

def res2pd(res, name, sessions=None):
    pd_res = []
    if sessions is None:
        sessions = [None] * len(res)
    for fs, s in zip(res, sessions):
        if s is None:
            sname = name
        else:
            sname = name + "_" + get_sess_names([s])[0]
        n_anim = -1
        ps = pd.DataFrame()
        for fname, f in zip(factors, fs):
            pf = pd.DataFrame()
            for anim in f:
                n_anim += 1
                if isinstance(anim, np.ndarray) or isinstance(anim, list):
                    pa = pd.DataFrame({sname:anim})
                else:
                    pa = pd.DataFrame({sname:[anim]})
                pa["animal"] = n_anim
                pf=pf.append(pa)
            pf["genotype"] = fname[1]
            pf["condition"] = fname[0]
            ps=ps.append(pf)
        pd_res.append(ps)
    return pd_res

def res_idx(r):
    return [k for k in r.keys() if not k in ["animal", "genotype", "condition"]]

def corr_test_2(res, name, sessions, covars=None, ctrl_covars=None, axes=None, show=True):
    
    res = res2pd(res, name, sessions)
    if covars is None:
        covars = []
    
    pair_wise_array = np.array([[0,0,1,1], #WT-Veh == WT-Glu
                                [0,1,0,1], #WT-Veh == Tg-Veh
                                [0,1,1,1], #WT-Veh == Tg-Glu
                                [0,0,-1,0], #Tg-Glu == Tg-Veh
                                [1,1,1,1], #WT-Veh
                                [1,1,0,0], #WT-Glu
                                [1,0,1,0], #Tg-Veh
                                [1,0,0,0], #Tg-Glu
                               ]) 
    pair_wise_labels = ['WT-Veh == WT-Glu',
                        'WT-Veh == Tg-Veh', 
                        'WT-Veh == Tg-Glu', 
                        'Tg-Glu == Tg-Veh',
                        'WT-Veh',
                        'WT-Glu',
                        'Tg-Veh',
                        'Tg-Glu',
                       ]
    
    if ctrl_covars is None:
        ctrl_covars = covars
    for r in res:
        ridx = res_idx(r)[0]
        assert np.all(res[0]["animal"].unique() == res[0]["animal"].unique())
    
        for var in ctrl_covars:
            if axes:
                ax = axes.pop()
            else:
                ax = plt.gca()
            for f, ft in zip(factors, factor_text):
                rd = r[np.logical_and(r["genotype"]==f[1], r["condition"]==f[0])].copy()
                dvar = var
                dridx = ridx
                if LATEX:
                    vdic = {k:verb(k) for k in rd.keys()}
                    rd = rd.rename(index=str, columns=verb)
                    dvar = verb(var)
                    dridx = verb(ridx)
                stats, stat_str = corr_str(rd[dvar], rd[dridx])
                l = ft + " " + stat_str
                sns.regplot(x=dvar, y=dridx, data=rd, ax=ax, label=l)
            sns.despine()
            leg = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
        formula = 'Q("'+ridx+'")' + " ~ genotype * condition"
        lm = smf.ols(formula, data=r).fit()
        pair_wise_res = lm.t_test(pair_wise_array).summary()
        pair_wise_res.insert_stubs(1, pair_wise_labels)
        if show:
            print(lm.summary())
            print(pair_wise_res)

            print()
        
        if ctrl_covars:

            formula = 'Q("'+ridx+'")' + " ~ " + " + ".join(["genotype * condition"] + ['Q("'+x+'")' for x in ctrl_covars])
            lm = smf.ols(formula, data=r).fit()

            pair_wise_res = lm.t_test(np.hstack([pair_wise_array, np.zeros((pair_wise_array.shape[0],len(ctrl_covars)))]))
            pair_wise_res = pair_wise_res.summary()
            pair_wise_res.insert_stubs(1, pair_wise_labels)
            if show:
                print(formula)
                print(lm.summary())
                print(pair_wise_res)
                print()
        return lm.summary(), pair_wise_res

def figsize(scale):
    fig_width_pt = 432.19737                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def set_fig_size(s):
    f = plt.gcf()
    f.set_size_inches(figsize(s))
    return f

def set_pgf_settings(base_font_size):
    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "text.latex.preamble": [r"\usepackage{siunitx}", ],
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "figure.figsize": figsize(1),     # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            r"\usepackage{siunitx}",
            ]
        }
    matplotlib.rcParams.update(pgf_with_latex)

def protected_replace(s, a, b):
    temp = "@@hahaha@@"
    return s.replace(b, temp).replace(a, b).replace(temp, b)
    
    
def verb(s): 
    if isinstance(s, str) and LATEX:
        ss = s.split("\n")
        nss = []
        for s in ss:
            s = protected_replace(s,"_","\\_")
            s = protected_replace(s,"%","\\%")
            s = protected_replace(s,"|", "$|$")
            nss.append(s)
        return "\n".join(nss) 
    else:
        return s

def bold_face(s):
    if isinstance(s, str) and LATEX:
        ss = s.split("\n")
        nss = []
        for s in ss:
            nss.append("\\textbf{{{}}}".format(s))
        return "\n".join(nss)
    else:
        return s
    
if LATEX:
    set_pgf_settings(fontsize)


# In[4]:


def make_fig(lh, lw, label_width=2, zoom=1/2.):
    label_width=lbw
    nh = sum([abs(x) for x in lh])
    nw = max([sum([(label_width if x is None else abs(x[0] if isinstance(x, tuple) else x)) for x in w]) for w in lw])
    all_fig = plt.figure(figsize=(zoom*nw, zoom*nh))
    gs = gridspec.GridSpec(nh,nw)
    labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    curr_y = 0
    axes = []
    i = 0
    for h in lh:
        if h < 0:
            curr_y += -h
            continue
        axes.append([])
        curr_x = 0
        for (j,w) in enumerate(lw[i]):
            if w is None:
                w = label_width
                ax = all_fig.add_subplot(gs[curr_y:curr_y+h, curr_x:curr_x+w])
                ax.axis("off")
                ax.text(0, 1.2, labels.pop(0), horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=fontsize+5, fontweight="bold", clip_on=False)
                curr_x += abs(w)
                continue
            elif isinstance(w, list) or isinstance(w, tuple):
                aw = w[0] 
                ah = h // w[1]
                share_type = "none"
                sharex = None
                sharey = None
                share_x = False
                share_y = False
                if len(w) == 3:
                    share_type = w[2]
                if share_type in ["share_x", "share_xy"]:
                    share_x = True
                if share_type in ["share_y", "share_xy"]:
                    share_y = True
                for k in range(w[1]):
                    if k==0:
                        share = None
                    elif k==1:
                        share = axes[i][-1]
                    if share_x:
                        sharex = share
                    if share_y:
                        sharey = share
                    axes[i].append(all_fig.add_subplot(gs[curr_y+ah*k:curr_y+ah*(k+1), curr_x:curr_x + aw], sharex=sharex, sharey=sharey))
                w = aw
            elif w > 0:
                axes[i].append(all_fig.add_subplot(gs[curr_y:curr_y+h, curr_x:curr_x+w]))
            

            curr_x += abs(w)
        curr_y += h
        i+=1

    for a in axes:
        for b in a:
            if b is None:
                continue
            b.spines["top"].set_visible(False)
            b.spines["right"].set_visible(False)
            b.yaxis.set_ticks_position('left')
            b.xaxis.set_ticks_position('bottom')
    res = all_fig.subplots_adjust(left=0,top=1,bottom=0,right=1, hspace=0, wspace=0)

    return all_fig, axes


# In[38]:


def factor2text(t):
    return t[1] + " + " +  ("$_{\u00A0}$TAT-Cntrl" if t[0] == "Veh" else r'TAT-$GluA2_{3Y}$')

factors = [('Veh', 'WT'), ('GluA2', 'WT'), ('Veh', 'Tg'), ('GluA2', 'Tg')]
factor_text = [factor2text(x) for x in factors]

def check_name(name):
    if name is None:
        return None, None
    return None, None
    fname = os.path.join("analysis", name + ".res.pkl")
    if not os.path.exists(fname):
        return fname, None
    with open(fname) as f:
        res = pickle.load(f)
    return fname, res


# In[31]:


#plotting and stats

def get_sess_names(a):
    sess = ['Home-pre-train', 'Ctx-pre-shock', 'Ctx-post-shock', 'Home-post-train', 'Ctx-test', 'Home-post-test', 'Ctx-pre-shock-1', 'Ctx-pre-shock-2', 'Ctx-test-5min', 'Ctx-test-3min', 'Ctx-shock', 'Ctx-test-last-5min']
    return [sess[i] for i in a]

def threshold(a, th):
    return (a-th).clip(0)


def pairwise_stats(r, factors, stat_func, stat_name, pos=.45, ax=None, show=True):
    if show and ax is None:
        ax = plt.gca()
    def stat_func_wrapped(*args, **kwargs):
        try:
            res = stat_func(*args, **kwargs)
        except Exception as e:
            return ["Stat Error:" + repr(e)]
        return res
        
    st_res = []
    for i in range(min(len(r), len(factors))):
        for j in range(i+1, min(len(r), len(factors))):
            st = [",".join(factors[k]) for k in (i,j)]
            st +=(list(stat_func_wrapped(r[i], r[j])))
            st_res.append(st)
    for y, stat in zip(np.linspace(pos, pos-.3, 6), st_res):
        if isinstance(stat[-1], str):#Error
            s = "{} vs {} {}".format(*stat)
            c = "r"
        else:
            s = r"{} vs {}, {}=\num{{{:.3e}}}, P=\num{{{:.2e}}}".format(*(stat[:2]+[stat_name]+stat[2:]))
            if stat[-1] < .001:
                c = "r"
            elif stat[-1] < .05:
                c = "k"
            else:
                c = "gray"
        if show:
            ax.text(1.1, y, s, color=c, fontsize=fontsize*5./8, transform=ax.transAxes)
    return st_res


def violin_plot(res, sessions, ax=None, ylabel=None, show=False, orient="v", show_legend=True, extra_legend=None, legend_pos="right", show_sample_size=True, bar_colors=face_colors, bar_edge_colors=line_colors, style="violin", ylim=None):
    if (isinstance(sessions[0], tuple) or isinstance(sessions[0], list)):
        sess_names = [" vs ".join(get_sess_names(x)) for x in sessions]
    elif isinstance(sessions[0], int):
        sess_names = [get_sess_names([x])[0] for x in sessions]
    if legend_pos == "top":
        loc = "lower left"
        lpos = (-.2, 1)
    elif legend_pos == "right":
        loc = "upper left"
        lpos = (1, 1)
    elif legend_pos == "top right":
        loc = "lower right"
        lpos = (1,1)
    elif isinstance(legend_pos, list) or isinstance(legend_pos, tuple):
        loc = "upper left"
        lpos = legend_pos
    for r, name in zip(res, sess_names):
        if ax is None:
            fig, ax = plt.subplots()
            #ax.set_title(name)
        if isinstance(r[0][0], list) or isinstance(r[0][0], np.ndarray):
            r = [np.hstack(x) for x in r]
        r = [np.array(x)[~np.isnan(x)] for x in r]
        means = [np.mean(x) for x in r]
        sterr = [np.std(x) / np.sqrt(len(x)) for x in r]
        ns = [np.array(x).shape[0] for x in r]
        x = np.arange(len(factors))
        if style == "violin":
            sns.violinplot(data=r, ax=ax, palette=face_colors, linecolor=line_colors, orient=orient)
        elif style == "box":
            sns.boxplot(data=r, ax=ax, palette=trace_colors,  orient=orient, notch=True, fliersize=.5, linewidth=1)
        sns.despine()
        #ax.set_xticks(x+.4)
        if show_sample_size:
            xlabels = [verb("{} ({})".format(f, n)) for f,n in zip(factor_text,ns)]
        else:
            xlabels = factor_text
        if orient=="v":
            ax.set_xticklabels([verb("{}\n({})".format(",".join(f), n)) for f,n in zip(factors,ns)])
            ax.set_ylabel(bold_face(verb(ylabel)))
            ax.set_ylim(ylim)
        elif orient=="h":
            ax.set_yticklabels([])
            ax.set_xlabel(bold_face(verb(ylabel)))
            ax.set_xlim(ylim)
        
        #stats, pairwise T-test
        legend = [patches.Patch(facecolor=c, edgecolor=ec, label=l, linewidth=line_width_bar) for c, ec, l in zip(bar_colors, bar_edge_colors, xlabels)]
        pairwise_stats(r, factors, ttest_ind, "T", show=False)
        if show_legend:
            l = ax.legend(handles = legend, loc=loc, bbox_to_anchor=(lpos[0], lpos[1]), handlelength=1)
            l.set_zorder(10)
            l.get_frame().set_linewidth(0.)
            ax.add_artist(l)
            if extra_legend is not None:
                extra_handles = extra_legend.legendHandles
                le = ax.legend(handles = extra_handles, labels=[t.get_text() for t in l.get_texts()], handlelength=1, 
                               loc=loc, bbox_to_anchor=(lpos[0]-.20, lpos[1]))
                le.get_frame().set_linewidth(0.)
                for t in le.get_texts():
                    t.set_color(transparent)
                ax.add_artist(le)
                le.set_zorder(0)
        

def box_plot(res, sessions, ax=None, ylabel=None, show=False):
    if (isinstance(sessions[0], tuple) or isinstance(sessions[0], list)):
        sess_names = [" vs ".join(get_sess_names(x)) for x in sessions]
    elif isinstance(sessions[0], int):
        sess_names = [get_sess_names([x])[0] for x in sessions]
    for r, name in zip(res, sess_names):
        if ax is None:
            fig, ax = plt.subplots()
            #ax.set_title(name)
        if isinstance(r[0][0], list) or isinstance(r[0][0], np.ndarray):
            r = [np.hstack(x) for x in r]
        r = [np.array(x)[~np.isnan(x)] for x in r]
        means = [np.mean(x) for x in r]
        sterr = [np.std(x) / np.sqrt(len(x)) for x in r]
        ns = [np.array(x).shape[0] for x in r]
        x = np.arange(len(factors))
        sns.boxplot(data=r, ax=ax, palette=face_colors) 
        sns.despine()
        #ax.set_xticks(x+.4)
        ax.set_xticklabels([verb("{}\n({})".format(",".join(f), n)) for f,n in zip(factors,ns)])
        ax.set_ylabel(bold_face(verb(ylabel)))
        
        #stats, pairwise T-test
        pairwise_stats(r, factors, ttest_ind, "T", show=False)


def d2ax(ax, data):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    lim = np.array([xlim, ylim])
    lim_min = lim[:,0]
    lim_max = lim[:,1]
    return (np.array(data) - lim_min) / (lim_max - lim_min)
    
    data_to_axis = ax.transData + ax.transAxes.inverted()
    return data_to_axis.transform(data)

def ax2d(ax, data):
    axis_to_data = ax.transAxes + ax.transData.inverted()
    return axis_to_data.transform(data)

def ax2f(ax, coords):
    axis_to_fig = ax.transAxes + ax.figure.transFigure.inverted()
    return axis_to_fig.transform(coords)

def ax2ax(ax1, ax2, coords):
    axis_to_axis = ax1.transAxes + ax2.transAxes.inverted()
    return axis_to_axis.transform(coords)


def sig_bar(ax, p1, p2, h, orientation="vertical", **kwargs):
    points = np.array([p1, p2, (h,h)])
    ps = d2ax(ax, points)
    if orientation=="horizontal":
        ps = d2ax(ax, points[:,::-1])
    x1, y1 = ps[0]
    x2, y2 = ps[1]
    hx, hy = ps[2]
    kwargs.update({"clip_on" : False,
                   "transform" : ax.transAxes
                   })
    if orientation=="horizontal":
        ax.plot((x1, hx), (y1, y1), **kwargs)
        ax.plot((hx, hx), (y1, y2), **kwargs)
        ax.plot((hx, x2), (y2, y2), **kwargs)
    else:
        ax.plot((x1,x1), (y1, hy), **kwargs)
        ax.plot((x1,x2), (hy, hy), **kwargs)
        ax.plot((x2,x2), (hy, y2), **kwargs)

   
def point_scale(ax):
    #how many points in 1 data unit, (x,y)
    xdisp_scale = ax.transData.transform((2,1))[0] - ax.transData.transform((1,1))[0]
    ydisp_scale = ax.transData.transform((1,2))[1] - ax.transData.transform((1,1))[1]
    disp_scale = np.array([xdisp_scale, ydisp_scale])
    dpi = ax.figure.dpi
    return disp_scale * 72. / dpi 

def save_fig(name, session, fig=None, transparent=False, axes=[], extra_artists=[],
             bbox_inches="tight", ftypes=["png","svg", "pgf"], update=False):
    arr_name = name.split(".")
    fname = "{}.{}".format(name, session)
    if fig is None:
        fig = plt.gcf()
    #[x.set_title("") for x in axes]
    for ft in ftypes:
        fn = fname + "." + ft
        if not update and os.path.exists(fn):
            continue
        if ft == "tex":
            try:
                tikz_save(fname+"."+ft, f)
            except:
                continue
        else:
            fig.savefig(fname+"."+ft, transparent=transparent, bbox_extra_artists=extra_artists, bbox_inches=bbox_inches, dpi=300)
            
def sig_bar_plot(ax, heights, sig_res=None, sig_res_mat=None, symbol="*", symbol_margin=fontsize*2, tick_length=3, sig_arr = None, sig_th=.01, reverse_y = False, factors=factors, xs=None, show_ns=False, type="sig", orientation="vertical"):
    if orientation == "vertical":
        ax_scale = point_scale(ax)[1]
    else:
        ax_scale = point_scale(ax)[0]
    smargin = abs(symbol_margin / ax_scale) 
    tlength = abs(tick_length / ax_scale)
    xmargin = 0
    th = sig_th
    symbol=bold_face(verb(symbol))
    fs = [",".join(x) for x in factors]
    if sig_res_mat is not None:
        mat = sig_res_mat
        mat[mat > th] = 0
    elif sig_res is not None:
        sig_res = [[fs.index(x[0]), fs.index(x[1]), x[-1]] for x in sig_res if x[-1] < th]
        mat = np.zeros((len(heights), len(heights)))
        for i,j,p in sig_res:
            mat[i,j] = p
            mat[j,i] = p
    #make sure Tg-Veh is first to show up on statsbar
    if sig_arr is None:
        order = np.hstack([2, np.argsort(-np.sum(mat > 0, axis=1))])
    else:
        order = [x[0] for x in sig_arr]
    heights = np.array(heights) + smargin
    is_sig = False
    if xs is None:
        xs = list(range(len(heights)))
    for i in order:
        if not show_ns:
            targets = np.where(mat[i]>0)[0]
        else:
            targets = np.where(mat[i]>=0)[0]
        if sig_arr:
            targets = [t for t in targets if (i,t) in sig_arr]
        if len(targets) == 0:
            continue
        is_sig = True
        h = -99999999
        func = np.max

        for t in targets:
            h = func([func(heights[min(t,i):(max(t, i)+1)])-smargin/2,heights[t],h]) + tlength
            
            if reverse_y:
                sig_bar(ax, (xs[i]+xmargin, -heights[i]-smargin/2.), (xs[t]+xmargin, -heights[t]), -h, color=stat_bar_color, orientation=orientation)
                ax.annotate(symbol, xy = (xs[t]+xmargin, -heights[t] - smargin/4.*3.2), fontsize = symbolfontsize, horizontalalignment="center", verticalalignment="center", orientation=orientation)
            elif mat[i, t] > 0:
                sig_bar(ax, (xs[i]+xmargin, heights[i]-smargin/2.), (xs[t]+xmargin, heights[t]), h, color=stat_bar_color, orientation=orientation)
                if orientation == "vertical":
                    xy = (xs[t]+xmargin, heights[t] - smargin/4.*3.2)
                elif orientation == "horizontal":
                    xy = (heights[t] - smargin/4.*3.2, xs[t]+xmargin)
                if mat[i, t] < sig_th:
                    this_symbol = symbol
                if mat[i, t] < sig_th * .2:
                    this_symbol += symbol
                if mat[i, t] < sig_th * .02:
                    this_symbol += symbol
                ax.annotate(this_symbol, xy, fontsize = symbolfontsize, horizontalalignment="center", verticalalignment="center", rotation=0 if orientation == "vertical" else 270)
            else:
                #h = max(heights[i], heights[t]) - smargin / 2
                sig_bar(ax, (xs[i]+xmargin, heights[i]-smargin/2.), (xs[t]+xmargin, heights[t]), h, color=stat_bar_color, orientation=orientation)
                ax.annotate("n.s.", xy=((xs[i] + xs[t])/2, h + smargin/2), xycoords="data", fontsize=fontsize-2, horizontalalignment="center", verticalalignment="center", clip_on=False, annotation_clip=False)

            #heights[t] = h + smargin/4.*3
        for t in range(min(i, np.min(targets)), max(i, np.max(targets))+1):
            heights[t] = h + smargin/4.*3
        mat[i, t] = 0
        mat[t,i] = 0
    if not is_sig:
        h = np.max(heights) - smargin / 3.
        sig_bar(ax, (0,h), (xs[-1],h), h, color=stat_bar_color)
        ax.annotate("n.s.", xy=(xs[-1]/2., h - smargin/4), xycoords="data", fontsize=fontsize-2, horizontalalignment="center", verticalalignment="center", clip_on=False, annotation_clip=False)

    return np.max(heights)-smargin/4.*3


def bar_plot(res, sessions, ylabel=None, show=True, stat_pos=.35, ax=None, extra_legend = None, sig_th=.05, sig_arr=None, show_stat=None, show_data=False, ylim=None, 
             save_as=None, fig_size=None, show_legend=True,legend_pos="right", show_sample_size=True, bar_colors=face_colors, bar_edge_colors=line_colors, 
             label_stat=True, alpha=1, paired=None, xlabel_rotation=0, bottom=None, hatch_colors=hatch_colors, orientation="vertical", error="se", show_ns=False, 
             sig_type="sig", extra_legend_pos=-.2, jitter_scale=.3, show_lines=False, plot_type="bar", baseline_label=None, sig_symbol="*", data_alpha=0.15,
             line_alpha=0.15, data_dot_style = "k.", ecap_ratio=0.5):
    
    linewidth = line_width_bar
    barwidth = 40/60.
    #ecap_ratio = .5
    group_margin = .5
    #data_dot_style = "k."
    data_line_style = "b-"
    #data_alpha = .15
    show_gca = False
    r_all = []
    if ax is None:
        show_gca = True
    
    
    if (isinstance(sessions[0], tuple) or isinstance(sessions[0], list)):
        sess_names = [" vs ".join(get_sess_names(x)) for x in sessions]
    elif isinstance(sessions[0], int):
        sess_names = [get_sess_names([x])[0] for x in sessions]
    elif isinstance(sessions[0], str):
        sess_names = sessions
    
    if legend_pos == "top":
        loc = "lower left"
        lpos = (-.2, 1)
    elif legend_pos == "right":
        loc = "upper left"
        lpos = (1, 1)
    elif legend_pos == "top right":
        loc = "lower right"
        lpos = (1,1)
    elif isinstance(legend_pos, list) or isinstance(legend_pos, tuple):
        loc = "upper left"
        lpos = legend_pos
        
    pt_transform = ax.transData
    ecap_width = pt_transform.transform([(0,0),(0,barwidth*ecap_ratio)])
    ecap_width = ecap_width / 22
    ecap_width = np.sqrt(np.sum((ecap_width[0] - ecap_width[1])**2))
    for i, r, sess, name in zip(range(len(sessions)), res, sessions, sess_names):
        if show_gca:
            ax = plt.gca()
            if fig_size is None:
                fs = figsize(0.8)
            else:
                fs = fig_size
            plt.gcf().set_size_inches(fs[0], fs[1], forward=True)
        if LATEX:
            name = verb(name)
        #ax.set_title(name)
        
        #offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        #pt_transform = ax.transData + offset
        
        if isinstance(r[0][0], list) or isinstance(r[0][0], np.ndarray):
            r = [np.hstack(x) for x in r]
        r = [np.array(x)[~np.isnan(x)] for x in r]
        means = [np.mean(x) for x in r]
        if error == "se":
            sterr = [np.std(x) / np.sqrt(len(x)) for x in r]
        elif error == "ci":
            #FIXME
            upper = [np.percentile(x, 95) - np.mean(x) for x in r]
            sterr = upper
        elif error == "std":
            sterr = [np.std(x) for x in r]

        ns = [np.array(x).shape[0] for x in r]
        xs = np.arange(len(factors))

        hatch = None
        if paired == "session":
            xs = xs + (len(xs) + group_margin) * i
            hatch = session_hatchset[i]
        elif paired == "condition":
            xs = xs * (len(sessions)+group_margin) + i
            hatch = hatchset[i]
        if bottom is None:
            low_limit = 0
        else:
            low_limit = bottom
            
        for n, m, e, x, c, ec, hc, rr, fn in zip(ns, means, sterr, xs, bar_colors, bar_edge_colors, hatch_colors, r, factor_text):
            if m >= low_limit:
                ye = [[0], [e]]
            else:
                ye = [[e], [0]]
            l = fn
            if show_sample_size:
                l+="({})".format(n)
            if LATEX:
                l = verb(l)
            if plot_type == "violin":
                v = ax.violinplot([rr], [x], showmedians=False, showextrema=False, points=20)
                for vp in v['bodies']:
                    vp.set_facecolor(c)
                    vp.set_edgecolor(ec)
                    vp.set_linewidth(2)
                    vp.set_alpha(1)
                b = ax.boxplot([rr], positions=[x], showcaps=False)
                for bm, bb in zip(b['medians'], b['boxes']):
                    c = bb.get_color()
                    bm.set_color(c)
                for bf in b['fliers']:
                    bf.set_markersize(1)
                    bf.set_markeredgewidth(0)
                    bf.set_markerfacecolor('k')
                    
            elif plot_type == "bar":
                if orientation == "vertical":
                    ax.bar([x], [m - low_limit], yerr=ye, color=c, edgecolor=ec, \
                            linewidth=linewidth, width=barwidth, \
                            alpha = alpha, \
                            error_kw=dict(\
                                ecolor=ec, \
                                capsize=ecap_width, \
                                linewidth=linewidth, \
                                markeredgewidth=linewidth,
                                # solid_capstyle="round",
                                alpha=alpha,
                                ), 
                            bottom=bottom,
                        label=l)
                elif orientation == "horizontal":
                    ax.barh([x], [m - low_limit], xerr=ye, color=c, edgecolor=ec, \
                            linewidth=linewidth, height=barwidth, \
                            alpha = alpha, \
                            error_kw=dict(\
                                ecolor=ec, \
                                capsize=ecap_width, \
                                linewidth=linewidth, \
                                markeredgewidth=linewidth,
                                # solid_capstyle="round",
                                alpha=alpha,
                                ), 
                            left=bottom,
                        label=l)
                if paired: #hatch
                    y_offset=.003
                    x_scale = .9
                    ax.bar([x], [m-y_offset-low_limit], hatch = hatch, facecolor='none', edgecolor=hc, linewidth=0, width=barwidth * x_scale, bottom=bottom)
            if show_data:
                rr = np.array(rr)
                try:
                    kde = gaussian_kde(rr / jitter_scale)
                    density = kde(rr / jitter_scale)
                except np.linalg.LinAlgError:
                    density = 0.01
                jitter = np.random.random(len(rr)) - 0.5 
                jitter = density * 2 * jitter * 0.2 * barwidth
                ax.plot(x + jitter, rr, data_dot_style, zorder=10, alpha=data_alpha)

        if show_sample_size == True:
            xlabels = [verb("{} ({})".format(f, n)) for f,n in zip(factor_text,ns)]
        elif show_sample_size == 'linebreak':
            xlabels = [verb("{} \n({})".format(f, n)) for f,n in zip(factor_text,ns)]
        else:
            xlabels = factor_text
        if orientation == "vertical":
            if bottom is not None:
                ax.spines['bottom'].set_position(('data', bottom))
            sns.despine()
            ax.set_xticks([])
            ax.set_xlim(-.5-(1-barwidth), len(res[0])-.5+(1-barwidth))
        elif orientation == "horizontal":
            if bottom is not None:
                ax.spines['left'].set_position(('data', bottom))
            sns.despine()
            ax.set_yticks([])
            ax.set_ylim(3+.5+(1-barwidth), -.5-(1-barwidth))

        if paired == "condition":
            xs = xs * (len(sessions)+group_margin) + i
            #ax.set_xlim(-.5-(1-barwidth), len(sessions) * (len(factors) + 1)- .5 - (1-barwidth)-.1)
            ax.set_xlim(-.5-(1-barwidth), (len(factors)-.5)*(len(res) + group_margin)+ (len(res)-1)/2)
            ax.set_xticks(np.arange(len(factors)) * (len(res) + group_margin) + (len(res)-1) / 2.)
            ax.tick_params(axis='x', which='major', pad=5)
            xticklabels = [s.replace(" + ", "\n") for s in xlabels]
            ax.set_xticklabels(xticklabels, rotation=xlabel_rotation)
            for tick in ax.xaxis.get_major_ticks():
                try:   # tick.label is depricated since matplotlib 3.8.0 (https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.8.0.html#unused-methods-in-axis-tick-xaxis-and-yaxis)
                    tick.label.set_verticalalignment('top')
                except AttributeError:
                    tick.label1.set_verticalalignment('top')
                
        if paired == "session":
            ax.set_xlim(-.5-(1-barwidth), len(sessions) * (len(factors) + 1)- .5 - (1-barwidth+group_margin))
            ax.set_xticks(np.arange(len(sessions)) * (len(factors) + group_margin) + (len(factors)-1) / 2.)
            ax.set_xticklabels(sessions)
        #ax.set_xticklabels(xlabels)
        if ylim is not None:
            if orientation == "vertical":
                ax.set_ylim(ylim)
            elif orientation == "horizontal":
                ax.set_xlim(ylim)
        
        ax.set_ylabel(bold_face(verb(ylabel)))
        
        #stats, pairwise T-test
        if show_stat is None:
            show_stat = show

        if not paired:
            stat_res = pairwise_stats(r, factors, ttest_ind, "T", pos=stat_pos, ax=ax, show=show_stat)
            
            #ax.set_autoscaley_on(False)
            if label_stat:
                if show_data or show_lines or plot_type == "violin":
                    heights = np.array([np.max(x) for x in r])
                else:
                    heights = np.array(means) + np.array(sterr)
                if bottom is not None:
                    heights = heights.clip(min=bottom)
                max_height = sig_bar_plot(ax, heights, stat_res, sig_arr=sig_arr, sig_th=sig_th, orientation=orientation, show_ns=show_ns, type=sig_type, symbol=sig_symbol)
                max_height_ax = d2ax(ax, [max_height, max_height])
                if orientation == "vertical":
                    max_height_ax = max_height_ax[1]
                else:
                    max_height_ax = max_height_ax[0]

                if max_height_ax > 1:
                    shrink_axis(ax, 1- 1./max_height_ax, orientation=orientation)
                if max_height_ax > 1:
                    lpos = list(lpos)
                    if orientation == "vertical":
                        lpos[1] = lpos[1] * max_height_ax
                    else:
                        lpos[0] = lpos[0] * max_height_ax
        if paired != "condition": 
            legend = [patches.Patch(facecolor=c, edgecolor=ec, label=l, linewidth=line_width_bar) for c, ec, l in zip(bar_colors, bar_edge_colors, xlabels)]

            l = None
            if show_legend:
                l = ax.legend(handles = legend, loc=loc, bbox_to_anchor=(lpos[0], lpos[1]), handlelength=1)
                l.get_frame().set_linewidth(0.)
                l.set_zorder(10)
                ax.add_artist(l)
                if extra_legend is not None:
                    extra_handles = extra_legend.legendHandles
                    le = ax.legend(handles = extra_handles, labels=[t.get_text() for t in l.get_texts()], handlelength=1, 
                                   loc=loc, bbox_to_anchor=(lpos[0]+extra_legend_pos, lpos[1]))
                    le.get_frame().set_linewidth(0.)
                    for t in le.get_texts():
                        t.set_color(transparent)
                    ax.add_artist(le)
                    le.set_zorder(0)
            
        

        if show_gca and save_as and l:
            l.remove()
            ll = ax.legend(loc=loc, bbox_to_anchor=lpos, handlelength=1)
            ll.get_frame().set_linewidth(0.)
            extra_artists = [ll]
            if not show_legend:
                extra_artists.pop()
                ll.set_visible(False)
            save_fig(save_as+".bar", sess, axes=[ax], extra_artists=extra_artists)

        if paired:
            r_all += r

    if paired == "condition":
        f_all = []
        rr_all = []
        xs = []
        x = 0
        for i in range(len(factors)):
            rr_all += r_all[i::len(factors)]
            for j in range(len(sessions)):
                f_all.append(list(factors[i]) + [str(j)])
                xs.append(x)
                x += 1
            x += group_margin
        
        if show_lines:
            ns = len(sessions)
            for f in range(len(factors)):
                sess_r = rr_all[f*ns:f*ns+ns]
                sess_xs = xs[f*ns:f*ns+ns] 
                for i in range(min([len(x) for x in sess_r])):
                    d = [x[i] for x in sess_r]
                    ax.plot(sess_xs, d, data_line_style, alpha=line_alpha)

        r_all = rr_all
        stat_res = pairwise_stats(r_all, f_all, ttest_ind, "T", pos=stat_pos, ax=ax, show=show_stat)
        mean_all = [np.mean(x) for x in r_all]
        sterr_all = [np.std(x) / np.sqrt(len(x)) for x in r_all]
        if label_stat:
            if show_data or show_lines or plot_type == "violin":
                heights = np.array([np.max(x) for x in r_all])
            else:
                heights = np.array(means) + np.array(sterr)
            if bottom is not None:
                heights = heights.clip(min=bottom)
            max_height = sig_bar_plot(ax, heights, stat_res, sig_arr=sig_arr, sig_th=sig_th, factors = f_all, xs=xs, show_ns=True, orientation=orientation, type=sig_type, symbol=sig_symbol)

            max_height_ax = d2ax(ax, [max_height, max_height])
            if orientation == "vertical":
                max_height_ax = max_height_ax[1]
            else:
                max_height_ax = max_height_ax[0]

            if max_height_ax > 1:
                shrink_axis(ax, 1- 1./max_height_ax, orientation=orientation)
            if max_height_ax > 1:
                lpos = list(lpos)
                if orientation == "vertical":
                    lpos[1] = lpos[1] * max_height_ax
                else:
                    lpos[0] = lpos[0] * max_height_ax

    if paired == "session":
        f_all = []
        xs = []
        x = 0
        for i in range(len(sessions)):
            for j in range(len(factors)):
                f_all.append(list(factors[j]) + [str(i)])
                xs.append(x)
                x += 1
            x += group_margin
                
        stat_res = pairwise_stats(r_all, f_all, ttest_ind, "T", pos=stat_pos, ax=ax, show=show_stat)
        mean_all = [np.mean(x) for x in r_all]
        sterr_all = [np.std(x) / np.sqrt(len(x)) for x in r_all]
        if label_stat:
            if show_data or show_lines or plot_type == "violin":
                heights = np.array([np.max(x) for x in r_all])
            else:
                heights = np.array(means) + np.array(sterr)
            if bottom is not None:
                heights = heights.clip(min=bottom)
            max_height = sig_bar_plot(ax, heights, stat_res, sig_arr=sig_arr, sig_th=sig_th, factors = f_all, xs=xs, show_ns=True, orientation=orientation, type=sig_type, symbol=sig_symbol)
            max_height_ax = d2ax(ax, [max_height, max_height])
            if orientation == "vertical":
                max_height_ax = max_height_ax[1]
            else:
                max_height_ax = max_height_ax[0]

            if max_height_ax > 1:
                shrink_axis(ax, 1- 1./max_height_ax, orientation=orientation)
            if max_height_ax > 1:
                lpos = list(lpos)
                if orientation == "vertical":
                    lpos[1] = lpos[1] * max_height_ax
                else:
                    lpos[0] = lpos[0] * max_height_ax

    if paired == "condition" and show_legend: 
        legend = [patches.Patch(facecolor='none', edgecolor="#3a3a3a", label=l, linewidth=line_width_bar, hatch=h) for l, h in zip(sess_names, hatchset)]
        lp = ax.legend(handles=legend, loc=loc, bbox_to_anchor=(lpos[0], lpos[1]))
        lp.get_frame().set_linewidth(0.)

    return ax

def hist_plot(data, normed=False, bins=50, cumulative=False, label=None, ax=None, **kwargs):
    if ax is None:
        ax=plt.gca()
    values, base = np.histogram(data, bins=bins)
    if normed:
        values = values.astype(float) / float(np.sum(values))
    if cumulative:
        values = np.cumsum(values)
    plot = ax.plot(base[:-1], values, label=label, **kwargs)
    return plot
'''
def get_bw(x):
    span = np.max(x) - np.min(x)
    grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.logspace(span*.0001, span*.005, 30)},
                    cv=10) # 20-fold cross-validation
    grid.fit(x[:, None])
    print(grid.best_params_)
    return grid.best_params_["bandwidth"]
'''
        
def dist_plot(res, sessions, xlabel=None, ylabel="Cumulative proportion", cumulative=True, show=True, stat_pos=.45,               ax=None, stat_ax=None, show_stats=False, save_as=None, xlim=None):
    
    edge_colors = colorset
    styles = line_styles
    
    if ax is None:
        show_gca = True
    else:
        show_gca = False
    xlabel = bold_face(verb(xlabel))
    ylabel = bold_face(verb(ylabel))
    
    if (isinstance(sessions[0], tuple) or isinstance(sessions[0], list)):
        sess_names = [" vs ".join(get_sess_names(x)) for x in sessions]
    elif isinstance(sessions[0], int):
        sess_names = [get_sess_names([x])[0] for x in sessions]
    for corrs, sess, name in zip(res, sessions, sess_names):
        if show_gca:
            ax = plt.gca()
        #ax.set_title(name)
        corrs = [np.hstack(x) for x in corrs]
        corrs = [x[~np.isnan(x)] for x in corrs]
        
        if not cumulative:
            ylabel = "Density"
            hists = [sns.distplot(corr, norm_hist=True,                                   hist=False, hist_kws={"histtype":"step","alpha":1,"linewidth":line_width_dist},                                  #kde_kws={"kernel":"Epa"},\
                                   label="{} ($n={}$)".format("-".join(f), corr[~np.isnan(corr)].shape[0])) for corr,f in zip(corrs, factors)]
        else:
            hists = [hist_plot(corr, ax=ax, normed=True, bins=50, cumulative=cumulative,                                label="{} ($n={}$)".format("-".join(f), corr[~np.isnan(corr)].shape[0]), color=ec,
                               linestyle=s, linewidth=line_width_dist) \
                     for corr, f, ec, s in zip(corrs, factors, edge_colors, styles)]
            ax.set_ylim([0,1])
        l = ax.legend(loc="lower right", bbox_to_anchor=(1,.02))
        l.set_visible(False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        sns.despine()

        #stats, pairwise KS-2sample
        if show_stats and stat_ax is None:
            stat_ax = ax
        if show_stats and stat_ax is not None:
            pairwise_stats(corrs, factors, ks_2samp, "K", pos=stat_pos, ax=stat_ax, show=True)
        if show_gca and save_as:
            save_fig(save_as+".dist", sess, axes=[ax], extra_artists=[l])

def set_label_size(ax, s):
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontsize(s)

def summ_plot(res, sessions, label=None, cumulative=True, axes=None, show=True, xlim=None, ylim=None, sig_arr=[(0,2),(2,3)], show_legend=True,
              save_as=None, more_wspace=0, show_sample_size=True, 
              sig_th=0.05, inset_style="bar", orientation="vertical"):
    ax_list = []
    for r,s in zip(res, sessions):
        if axes is None:
            gca = True
            fs = figsize(1.)
            f, (ax1, ax2) = plt.subplots(1,2, figsize=(fs[0], fs[1]))
            f.subplots_adjust(wspace=fig_wspace + more_wspace)
            b = ax2.get_position()
            pb = b.get_points()
            pb[1][1] -= .2 * (pb[1][1] - pb[0][1])
            b.set_points(pb)
            ax2.set_position(b)
            show_stats = True
        else:
            gca = False
            show = False
            show_stats = False
            ax1 = axes.pop(0)
            ax2 = axes.pop(0)
            legend_pos = "right"
            inset = False
            if isinstance(ax2, list) or isinstance(ax2, np.ndarray): #inset
                ax2_pos = ax2f(ax1, ax2)
                ax2 = ax1.figure.add_axes([ax2_pos[0][0],
                                     ax2_pos[0][1],
                                     ax2_pos[1][0] - ax2_pos[0][0],
                                     ax2_pos[1][1] - ax2_pos[0][1]])
                inset = True
                #set_label_size(ax2, inset_label_size)
                legend_pos = ax2ax(ax1, ax2, (1.1, 1)).tolist()


            show_stats = False
        if save_as:
            show_stats = False
        dist_plot([r], [s], xlabel=label, cumulative=cumulative, show=False, stat_pos=.35, ax=ax1, stat_ax=None, show_stats=show_stats, xlim=xlim)
        dist_legend = ax1.legend_
        if inset_style == "bar":
            bar_plot([r], [s], ylabel="", show=False, ax=ax2, extra_legend=dist_legend, sig_arr=sig_arr, show_stat= show_stats and (save_as is None), legend_pos=legend_pos, ylim=ylim, show_legend=show_legend, show_sample_size=show_sample_size, sig_th=sig_th, orientation=orientation)
        elif inset_style in ["violin", "box"]:
            violin_plot([r], [s], ax=ax2, show=False, ylabel="", orient="h", show_legend=show_legend, ylim=None, extra_legend=dist_legend, legend_pos=legend_pos, style=inset_style, bar_colors=trace_colors, bar_edge_colors=[gray]*4)
            ax2.tick_params(axis='x', bottom=True)

        if inset:
            pass
            #ax2.set_ylabel("")
        if gca and save_as:
            save_fig(save_as+".summ", s, fig=f, axes=[ax1, ax2], bbox_inches="tight")# extra_artists= [t])
        ax_list += [ax1, ax2]
    return ax_list
    
        
        

def r_2samp(x, y):
    rs = [pearsonr(*m)[0] for m in [x,y]]
    ms = [np.log((1+r)/(1-r))/2 for r in rs]
    ns = [m[0].shape[0] for m in [x,y]]
    stesqs = [1/(n-3) for n in ns]
    z = (ms[1] - ms[0]) / np.sqrt(sum(stesqs))
    p = norm.sf(abs(z)) * 2
    return (z, p)



        
def corr_plot(res, sessions, xlabel, ylabel, ax=None, xlim=None, ylim=None, size=16, logx=False, show_stat=True):
    def corr_str(x,y):
        s = "R={:.3f}, P={:.3f}".format(*pearsonr(x,y))
        if LATEX:
            s = verb(s)
        return s
    xlabel = bold_face(verb(xlabel))
    ylabel = bold_face(verb(ylabel))
    if (isinstance(sessions[0], tuple) or isinstance(sessions[0], list)):
        sess_names = [get_sess_names(x) for x in sessions]
    gca = False
    if ax is None:
        gca = True
    for corrs, name in zip(res, sess_names):
        if gca:
            ax = plt.gca()
        if logx:
            ax.set_xscale('log')
        #plt.title(name)
        corrs = [np.hstack(x) for x in corrs]
        corrs = [x[:,~np.any(np.isnan(x),axis=0)] for x in corrs]
        for corr, f, c in zip(corrs, factor_text, colorset):
            if show_stat:
                lbl = "{} (n={}) {}".format(f, corr.shape[1], corr_str(corr[0], corr[1]))
            else:
                lbl = "{} ({})".format(f, corr.shape[1])
            cps = sns.regplot(corr[0], corr[1], ax=ax,
                               label=lbl,
                               color=c,
                               scatter_kws = {"s":size},
                               logx=logx)
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1,1), handletextpad=0)
        stat_corrs = corrs
        if logx:
            stat_corrs = [(np.log(corr[0]), corr[1]) for corr in corrs]
        rp = [pearsonr(corr[0], corr[1])[1] for corr in stat_corrs]
        if show_stat:
            for l,p in zip(leg.get_texts(), rp):
                if p < .001:
                    l.set_color("red")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        #stats, Fisher z-transformation -> pairwise 2-sample Z-test
        if show_stat:
            pairwise_stats(stat_corrs, factors, r_2samp, "Z", ax=ax, show=True)
        #plt.savefig("plots/"+"-".join(name)+"-"+xlabel.replace("/","-")+"-"+ylabel.replace("/","-")+".corr.svg")
    return ax
        
def draw_box(ax, pts, **kwargs):
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    ax.plot([x0, x1], [y0, y0], transform=ax.transAxes, clip_on=False, **kwargs)
    ax.plot([x1, x1], [y0, y1], transform=ax.transAxes, clip_on=False, **kwargs)
    ax.plot([x1, x0], [y1, y1], transform=ax.transAxes, clip_on=False, **kwargs)
    ax.plot([x0, x0], [y0, y1], transform=ax.transAxes, clip_on=False, **kwargs)

def box_insets(ax, inset):
    data_ax_coords = d2ax(ax, inset['data_coords'])
    ax_coords = inset['axis_coords']
    loc = inset['loc']
    kwargs = inset.get('kwargs', {'color':"r", 
                                  'linestyle':"--",
                                  'alpha':.5,
                                  })
    loc_margin = inset.get('loc_margin', [(0, 0), (0,0)])
    draw_box(ax, data_ax_coords, **kwargs)
    #draw_box(ax, ax_coords, **kwargs)
    for l in zip(*loc):
        i1,j1 = l[0] // 2, l[0] % 2
        i2,j2 = l[1] // 2, l[1] % 2
        ax.plot([data_ax_coords[i1][0]+loc_margin[0][0], 
                    ax_coords[i2][0] + loc_margin[1][0]],
                [data_ax_coords[j1][1] + loc_margin[0][1],
                    ax_coords[j2][1] + loc_margin[1][1]], 
                transform = ax.transAxes, clip_on=False, **kwargs)


def trace_plot(res, sessions, xs=None, ax=None, ylabel=None, refs=None, per_animal=False, show_err=False, show_n_trace=None, stat_refs=[], events=[0], title=None, save_as=None, xlim=None, ylim=None, fig_size=None, inset=None, show_legend=True, trace_colors=trace_colors):#refs = [(cond, res, sess)], #stat_ref=[(slice_ref, slice_test)]
    color = trace_colors
        
    gca = False
    if ax is None:
        gca = True
        show_gca=True
    else:
        gca = False
        show_gca=False
    if (isinstance(sessions[0], tuple) or isinstance(sessions[0], list)):
        sess_names = [" vs ".join(get_sess_names(x)) for x in sessions]
    elif isinstance(sessions[0], int):
        sess_names = [get_sess_names([x])[0] for x in sessions]
    
    for sdata, sname, sess in zip(res, sess_names, sessions):
        if gca:
            ax = plt.gca()
            f = plt.gcf()
            if fig_size is None:
                fig_size = figsize(1)
        
        if refs: 
            ref_r = [(cond, ref_res[ref_sessions.index(sess)], ref_sessions) for cond, ref_res, ref_sessions in refs]
        else:
            ref_r = None
       
        if title is None:
            title = sname
        if title:
            title = verb(title)
            #plt.title(title)
            
        ms = []
        stes = []
        if per_animal:
            sdata = [np.vstack([np.nanmean(y.astype(float), axis=0) for y in x]) for x in sdata]
        else:
            sdata = [np.vstack(x) for x in sdata]
        ms = [np.nanmean(x, axis=0) for x in sdata]
        stds = [np.nanstd(x, axis=0) for x in sdata]
        ns = [np.sum(~np.isnan(x), axis=0) for x in sdata]
        stes = [std/np.sqrt(n) for std, n in zip(stds, ns)]
        ci_tops = [m + ste for (m, ste) in zip(ms, stes)]
        ci_bottoms = [m - ste for (m, ste) in zip(ms, stes)]
        y_top = np.max(ci_tops)
        y_bottom = np.min(ci_bottoms)
        if xs is None:
            xs = np.arange(ms[0].shape[0])
        if show_n_trace is None:
            for c, f, m, ci_top, ci_bottom, n, i in zip(color, factor_text, ms, ci_tops, ci_bottoms, ns, range(len(ms))):
                ax.plot(xs, m, ls="solid", c=c, label=f, alpha=.8)
                if per_animal or show_err:
                    ax.fill_between(xs, ci_top, ci_bottom, facecolor=c, alpha=.2)
                ax.set_ylabel(bold_face(verb(ylabel)))
            
            for idx_ref, idx_test in stat_refs:
                mean = np.mean(m[idx_ref])
                pop_std = np.std(m[idx_ref])
                pop_n = (m[idx_ref]).size
                
                ste = ci_top - m
                ts = ((m - mean)/ste)[idx_test]
                tn = n[idx_test]
                
                ps = scipy.stats.t.sf(-ts, tn-1)

                
                txs = xs[idx_test]
                n_test = txs.size

                sig = ps < .05 / n_test * 2
                counter = 0
                maxlen = -1
                for s in ps:
                    if s < .01:
                        counter += 1
                    else:
                        maxlen = max(maxlen, counter)
                        counter = 0
                maxlen = max(maxlen, counter)
                txs = txs[sig]
                yb, yt = ax.get_ylim()
                y = y_top + (6-i) *(y_top-y_bottom) * .03
                tys = np.zeros_like(txs)+y
                ax.plot(txs, tys, c=c, marker=".", linestyle="none", markerfacecolor=c, markersize=4)
            
            if ref_r:
                for cond, r, _ in ref_r:
                    r = r[i]
                    rm = np.mean(np.hstack(r))
                    if cond:
                        rxs = xs[cond(xs)]
                    else:
                        rxs = xs
                    ax.plot(rxs, np.ones_like(rxs)*rm, c=c, ls="dashed")
               
            #ax.plot(xs, -n * .0005, ls="solid", c=c, label=f)
            
        else:
            for c, f, ys, n, i in zip(color, factor_text, sdata, ns, range(len(ms))):
                rand = np.random.choice(ys.shape[0], show_n_trace, replace=False)
                for y in ys[rand]:
                    ax.plot(xs, y, ls="solid", c=c, alpha=.5)
        
        for e in events:
            ax.axvline(x=e, color='#ffcc00', linestyle="dashed")
        if show_legend:
            l = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
            l.get_frame().set_linewidth(0.)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xlabel(bold_face("Time before freezing (s)"))
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if inset:
            dc = inset["data_coords"]
            ac = inset["axis_coords"]
            ac_pos = ax2f(ax, ac)
            axins = ax.figure.add_axes([ac_pos[0][0],
                                     ac_pos[1][1],
                                     abs(ac_pos[1][0] - ac_pos[0][0]),
                                     abs(ac_pos[0][1] - ac_pos[1][1])])
            trace_plot(res=res, sessions=[sess], ax=axins, xs=xs, ylabel=None, refs=None, per_animal=per_animal, show_err=show_err, show_n_trace = show_n_trace, stat_refs = [], events = events, title=None, save_as=None, xlim=[dc[0][0],dc[1][0]], ylim=[dc[1][1],dc[0][1]], inset=None, fig_size=None, show_legend=False, trace_colors=trace_colors)
            box_insets(ax, inset)
            axins.set_ylabel("")
            axins.set_xlabel("")
            set_label_size(axins, inset_label_size)

        
        if show_gca and save_as:
            save_fig(save_as+".trace", sess, axes=[ax], extra_artists=[l])
    if inset:
        return [ax, axins]
    else:
        return [ax]

            
def show_pairwise_stats(r, factors, stat_func, stat_name):
    def stat_func_wrapped(*args, **kwargs):
        try:
            res = stat_func(*args, **kwargs)
        except Exception as e:
            return ["Stat Error:" + repr(e)]
        return res
        
    st_res = []
    for i in range(len(factors)):
        for j in range(i+1, len(factors)):
            st = [",".join(factors[k]) for k in (i,j)]
            st +=(list(stat_func_wrapped(r[i], r[j])))
            st_res.append(st)
    for y, stat in zip(np.linspace(.45, .15, 6), st_res):
        if isinstance(stat[-1], str):#Error
            s = "{} vs {} {}".format(*stat)
            c = "r"
        else:
            s = "{} vs {}, {}={:.3e}, P={:.2e}".format(*(stat[:2]+[stat_name]+stat[2:]))
            if stat[-1] < .001:
                c = "r"
            elif stat[-1] < .05:
                c = "k"
            else:
                c = "gray"
        plt.figtext(.95, y, s, color=c)

def get_stats(r):
    all_stats = corr_test_2([r], 'temp', [4], covars=[], show=False)
    omnibus_P = float(all_stats[0].tables[0].data[3][3])
    return omnibus_P

def line_plot(res, ax=None, show=False, ylabel="% freezing", xlabel="Test time (minute)", xs=None, xlim=None, ylim=None, show_legend=True, xlog=False, show_sig=False):
    if ax is None:
        ax = plt.gca()
    means = np.array([[np.nanmean(x) for x in r] for r in res])
    sterr = np.array([[np.nanstd(x) / np.sqrt(len(x)) for x in r] for r in res])
    height = np.array([np.max([np.nanmean(x) + np.nanstd(x) / np.sqrt(len(x)) for x in r]) for r in res])
    if xs is None:
        xs = np.arange(means.shape[0])+1
    ns = np.array([np.array(x).shape[0] for x in res[0]])
    plots = [ax.errorbar(xs, m, label=f, yerr=e, color=lc, linestyle=ls)              for (m,f,e,lc,ls) in zip(means.T, factor_text, sterr.T,colorset, line_styles)]
    sns.despine()
    ax.set_xlabel(bold_face(verb(xlabel)))
    ax.set_ylabel(bold_face(verb(ylabel)))
    if show_legend:
        l = ax.legend(loc="upper left", bbox_to_anchor=(1.05,1))
        l.get_frame().set_linewidth(0.)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if show_sig:
        sigs = np.array([get_stats(x) for x in res])
        asterisk_coords = np.vstack([xs, height])
        asterisk_coords = d2ax(ax, asterisk_coords.T)
        asterisk_coords = asterisk_coords + np.array([0, .05])
        asterisk_coords = asterisk_coords[sigs < .05]
        for c in asterisk_coords:
            ax.annotate("*", xy=c, xycoords="axes fraction", fontsize=fontsize, horizontalalignment="center", verticalalignment="center", clip_on=False, annotation_clip=False)
    if xlog:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_ticks([.2, .5, 1])
    #show_pairwise_stats(res, factors, ttest_ind, "T")
    #ax.set_xlim(0, means.shape[0] + 1)
            


# In[8]:
def decide(pred_res):
    if pred_res["probability"]:
        return pred_res["classes"][np.argmax(pred_res["y_res"], axis=1)]
    else:
        return pred_res["y_res"]

def map_res(f, res):
    return [[[f(a) for a in fa if a is not None and f(a) is not None] for fa in s] for s in res]

def zip_sess(sess):
    return [list(zip(*s)) for s in zip(*sess)]


def get_confusion_matrix(res):
    def fcm(x):
        y_res = decide(x)
        cm = confusion_matrix(x["y"], y_res)
        if cm.size == 1 or np.any(cm/np.sum(cm) > .99):
            #print(cm)
            #cm = np.array([[1,0],[0,0]])
            cm = None
        return cm
    return map_res(fcm, res)

def get_recall_nfreeze(res):
    return map_res(lambda x: x[0][0]/(x[0][0]+x[0][1]), get_confusion_matrix(res))


def get_accuracy(res):
    return map_res(lambda x: (x[0][0]+x[1][1])/np.sum(x), get_confusion_matrix(res))

def get_balanced_accuracy(res):
    return map_res(lambda x: (x[0][0]/(x[0][0]+x[0][1]) + x[1][1]/(x[1][1]+x[1][0]))/2., get_confusion_matrix(res))

def get_precision(res):
    return map_res(lambda x: x[1][1]/(x[0][1]+x[1][1]), get_confusion_matrix(res))

def get_recall(res):
    return map_res(lambda x: x[1][1]/(x[1][0]+x[1][1]), get_confusion_matrix(res))

def get_specificity(res):
    return map_res(lambda x: x[0][0]/(x[0][1]+x[0][0]), get_confusion_matrix(res))

def get_informedness(res):
    return map_res(lambda x: x[1][1]/(x[1][0]+x[1][1]) + x[0][0]/(x[0][1]+x[0][0])-1, get_confusion_matrix(res))

def get_f1(res):
    return map_res(lambda x: x[1][1]*2/(np.sum(x) - x[0][0]+x[1][1]), get_confusion_matrix(res))


# In[9]:


def change_point_plot(cps, sess, save_as=None, ylim=None, ax=None, extra_legend=None, show_legend=True, bottom=None, ngroups=None, bar_colors=face_colors, bar_edge_colors=line_colors, trace_colors=trace_colors, plot_type="bar", data=None):
    linewidth = line_width_bar
    xlabels = factor_text
    barwidth = 40/60.
    ecap_ratio = .5
    loc = "upper left"
    lpos = (1, 1)
    pt_transform = ax.transData
    ecap_width = pt_transform.transform([(0,0),(0,barwidth*ecap_ratio)])
    ecap_width = ecap_width / 11
    ecap_width = np.sqrt(np.sum((ecap_width[0] - ecap_width[1])**2))
    if ax is None:
        ax = plt.gca()
    if ngroups is None:
        xs = np.arange(len(factors))# + (1 - barwidth)/2
        ax.set_xlim(-.5-(1-barwidth), len(cps)-.5+(1-barwidth))
    else:
        xs = np.arange(ngroups)
        ax.set_xlim(-.5-(1-barwidth), ngroups-.5+(1-barwidth))
    for i, x, cp, ec, c, fn in zip(range(len(xs)), xs, cps, bar_edge_colors, bar_colors, factor_text):
        if bottom is not None and cp[0] < bottom:
            yerr = [[0], [cp[1]]]
        else:
            yerr = [[cp[1]], [0]]
        if plot_type == "bar":
            ax.bar([x], [cp[0]], yerr=yerr, color=c, edgecolor=ec, linewidth=linewidth, width=barwidth, error_kw=dict(ecolor=ec, capsize=ecap_width, linewidth=linewidth, markeredgewidth=linewidth), label=fn, bottom=bottom)
        elif plot_type == "violin":
            v = ax.violinplot([data[i]], [x], showmedians=False, showextrema=False, points=20)
            for vp in v['bodies']:
                vp.set_facecolor(c)
                vp.set_edgecolor(ec)
                vp.set_linewidth(2)
                vp.set_alpha(1)
            b = ax.boxplot([data[i]], positions=[x], showcaps=False)
            for bm, bb in zip(b['medians'], b['boxes']):
                c = bb.get_color()
                bm.set_color(c)
            for bf in b['fliers']:
                bf.set_markersize(1)
                bf.set_markeredgewidth(0)
                bf.set_markerfacecolor('k')
    
    ax.axhline(y=0., color='#ffcc00', linestyle="dashed")
        
    if show_legend:
        legend = [patches.Patch(facecolor=c, edgecolor=ec, label=l, linewidth=line_width_bar) for c, ec, l in zip(bar_colors, bar_edge_colors, xlabels)]
        l = ax.legend(handles = legend, loc=loc, bbox_to_anchor=(lpos[0]+.15, lpos[1]), handlelength=1)
        l.get_frame().set_linewidth(0.)

        l.set_zorder(10)
        ax.add_artist(l)
        if extra_legend is not None:
            extra_handles = extra_legend.legendHandles
            le = ax.legend(handles = extra_handles, labels=[t.get_text() for t in l.get_texts()], handlelength=1, 
                           loc=loc, bbox_to_anchor=lpos)
            le.get_frame().set_linewidth(0.)
            for t in le.get_texts():
                t.set_color(transparent)
            ax.add_artist(le)
            le.set_zorder(0)
    #ll = ax.legend(loc=loc, bbox_to_anchor=lpos, fontsize=fontsize-2, handlelength=1)
    ax.set_ylabel(bold_face(verb("Time to freezing (s)")))
    if ylim:
        ax.set_ylim(ylim)
    sns.despine()
    ax.set_xticks([])
    if bottom is not None:
        ax.spines['bottom'].set_position(('data', bottom))
    if save_as:
        save_fig(save_as+".bar", sess, axes=[ax])

def title_plot(text, pos, ax, va="center", ha="center", rot='horizontal', show_data=True):
    ax.text(pos[0], pos[1], text, fontsize = labelfontsize, horizontalalignment=ha, verticalalignment=va, transform=ax.transAxes, clip_on=False, weight="bold", rotation=rot)
    if not show_data:
        ax.set_axis_off()

def confusion_matrix_plot_1(res, axes, title=None, cmap='jet'):
    for r, ax in zip(res, axes):
        r = r / np.sum(r, axis=(1,2))[:,None,None]
        image_plot(np.mean(r, axis=0), ax, vmin=0, vmax=1, cmap=cmap) 
        ax.set_axis_off()
    if title:
        title_plot(title, (.5, 1), ax=axes[0])

def confusion_matrix_plot(reslist, ax, titles, cmap='jet', margin_x=.1, margin_y=.1, vmin=None, vmax=None, title_rotation=0):
    s = 100
    py = int(s * 4 + 3 * margin_y * s)
    px = int(s * len(reslist) + (len(reslist) - 1) * margin_x * s)
    sy = int(s + s * margin_y)
    sx = int(s + s * margin_x)
    hs = s // 2
    cm_map = np.empty((py, px))
    cm_map[:,:] = np.nan
    for i, res in enumerate(reslist):
        xpos = i * sx
        for j, r in enumerate(res[0]):
            ypos = j * sy
            r = np.array(r)
            r = r / np.sum(r, axis=(1,2))[:,None,None]
            r = np.mean(r, axis=0)
            cm_map[ypos:ypos+hs,xpos:xpos+hs] = r[0,0]
            cm_map[ypos+hs:ypos+s, xpos:xpos+hs] = r[1,0]
            cm_map[ypos:ypos+hs,xpos+hs:xpos+s] = r[0,1]
            cm_map[ypos+hs:ypos+s, xpos+hs:xpos+s] = r[1,1]
    ax.imshow(cm_map, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
    ax.set_xticks(range(hs, px, sx))
    ax.set_yticks(range(hs, py, sy))
    ax.set_yticklabels(factor_text)
    if title_rotation == 0:
        ha = "center"
    elif title_rotation < 0:
        ha = "left"
    elif title_rotation > 0:
        ha = "right"
    ax.set_xticklabels(titles, rotation=title_rotation, ha=ha)



def image_plot(img, ax, vmin=None, vmax=None, cmap='jet',
               interpolation='nearest'):
    ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, 
                    interpolation=interpolation)

def cbar_plot(ax, vmin, vmax, steps=600, orient="virtical", cmap='jet', ylabel=""):
    nticks = 6
    cbar_data = np.linspace(vmin, vmax, steps)[::-1]
    cbar_img = np.empty((steps, steps // 6))
    cbar_img[:,:] = cbar_data[:, None]
    if orient == 'virtical':
        pass
    elif orient == 'horizontal':
        cbar_img = cbar_img.T
    ax.imshow(cbar_img, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.yaxis.tick_right()
    ax.set_yticks(np.linspace(0, steps, nticks)[::-1])
    ax.set_yticklabels(["{:.1f}".format(x) for x in np.linspace(vmin, vmax, nticks)])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_position("right")
    ax.set_xticks([])
    


def mc_read_stat(mcs):
    return mcs
   
def bayes_stats(res, sessions, mc_res, save_as=None, ylim=None, axes=None, inset=None, show_legend=True, show_sig=True, ylabel="Accuracy", sig_th=.01,
        sig_type="ns", bottom=None, sig_arr=[(0,2),(2,3)], stat_res_arr=None, bar_colors=face_colors, bar_edge_colors=line_colors, trace_colors=trace_colors, alt_legend_coord=np.array([3.91, .38]), r_sq_data=None, r_sq_axis=None):
    
    taus, dtau_idx, dtaus, a, selector = mc_read_stat(mc_res)
    taus = np.array(taus)
    mt = taus.mean(axis=1)
    letau = mt - np.percentile(taus, 2.5, axis=1) 
    retau = np.percentile(taus, 97.5, axis=1) - mt
    tau_stats = np.vstack([mt, letau, retau]).T
    for i in range(len(res[0])):
        us, cs = np.unique(selector[i], return_counts=True)
        if np.any(us==0) and (not np.any(us==1) or cs[1]  < cs[0]*10.):
            tau_stats[i] = np.nan
            
    if axes is None:
        gca = True
        fs = figsize(1)
        f, (ax1, ax2) = plt.subplots(1,2, figsize=(fs[0], fs[1]), gridspec_kw={"width_ratios":(2,1)})
        f.subplots_adjust(wspace=fig_wspace+.1)
        b = ax2.get_position()
        pb = b.get_points()
        pb[1][1] -= .2
        b.set_points(pb)
        ax2.set_position(b)
    else:
        gca = False
        show = False
        ax1 = axes.pop(0)
        ax2 = axes.pop(0)
        '''
        b = ax2.get_position()
        pb = b.get_points()
        pb[1][1] -= .2 * (pb[1][1] - pb[0][1])
        b.set_points(pb)
        ax2.set_position(b)
        '''
        show_stats = False
    
    #change_point_trace_plot(res, sessions, tau_stats/20-np.array([10,0,0]))
    ax, axins = trace_plot(res, [sessions[0]], xs=np.arange(-window, window, 1./frame_rate), ylabel=ylabel, title=[],  xlim=[-window, 5], ax=ax1, ylim=[0, 1], inset=inset, trace_colors=trace_colors)
    for r, ts in zip(res[0], tau_stats):
        r1 = np.vstack(r)
        x = ts
        if not np.isnan(x[0]):
            y = np.nanmean(r1[:, int(round(x[0]-x[1])):int(round(x[0]+x[2]))])
            x = x/20.
            x[0] -= 10
            ax.plot([x[0]],[y],"k.",zorder=100, markersize=2)
            axins.plot([x[0]],[y],"k.",zorder=100, markersize=4)
            ax.errorbar([x[0]],[y],  xerr=[[x[1]],[x[2]]], lw=1, color="k", zorder=100)
            axins.errorbar([x[0]], [y],  xerr=[[x[1]],[x[2]]], lw=1, color="k", zorder=100)

    trace_legend = ax1.legend_
    trace_legend.get_frame().set_linewidth(0.)
    trace_legend.set_visible(False)
    change_point_plot(-tau_stats/20.+np.array([10,0,0]), sessions[0], ylim=ylim, ax=ax2, extra_legend=trace_legend, show_legend=show_legend, bottom=bottom, ngroups=len(res[0]), bar_colors=bar_colors, bar_edge_colors=bar_edge_colors, trace_colors=trace_colors, plot_type="violin", data=[-t/20.+10 for t in taus])
    
    error_bar_len = np.array([.1, 0])
    text_margin = np.array([.07, 0])
    legend_ax_coords = np.vstack([alt_legend_coord - error_bar_len / 2.,
                             alt_legend_coord + error_bar_len / 2.,
                             alt_legend_coord + error_bar_len / 2 + \
                                     text_margin])
    center = np.mean(legend_ax_coords[:2], axis=0)
    ax.plot([center[0]], [center[1]], "k.", transform=ax.transAxes, clip_on=False)
    ax.plot(legend_ax_coords[:2,0], legend_ax_coords[:2,1], "k-", transform=ax.transAxes, clip_on=False, lw=1)
    ax.text(legend_ax_coords[2,0], legend_ax_coords[2,1], "Change point", transform=ax.transAxes, va="center", clip_on=False, fontsize=12)

    if r_sq_axis and r_sq_data:
        violin_plot(r_sq_data, sessions, ax=r_sq_axis, show=False, ylabel="Goodness of fit (R^2)", show_legend=False)
        r_sq_axis.set_xticks([])
        r_sq_axis.set_ylim([.9, 1.])

    
    if gca and save_as:
        save_fig(save_as+".summ", sessions, fig=f, axes=[ax1, ax2], bbox_inches="tight")# extra_artists= [t])
    for i in range(len(res[0])):
        us, cs = np.unique(selector[i], return_counts=True)
        n = len(us)
    
    dtau_null = len(taus[0])/200.
    stat_res = np.zeros((len(res[0]), len(res[0])))
    for idx, dtau in zip(dtau_idx, dtaus):
        if idx[0] >= len(res[0]) or idx[1] >= len(res[0]):
            continue
        bf = np.sum(dtau == 0) / dtau_null
        if bf != 0:
            bf = min(bf, 1./bf)
        else:
            bf = 1. / dtau_null
        stat_res[idx[0], idx[1]] = bf 
        stat_res[idx[1], idx[0]] = bf
    heights = 10-(tau_stats[:,0] - tau_stats[:,1])/20.
    nan_height = np.where(np.isnan(heights))[0]
    for nh in nan_height:
        ax2.annotate("#", xy = (nan_height, .3), fontsize = symbolfontsize, horizontalalignment="center", verticalalignment="center")

    heights[np.isnan(heights)] = 0
    # heights[heights < 0] = 0
    '''
    if stat_res.shape[0] == 3:
        stat_res[0,1]=0
        stat_res[1,0]=0
    '''
    if stat_res_arr is not None:
        stat_res = stat_res_arr
    if show_sig:
        sig_bar_plot(ax2, heights ,sig_res_mat=stat_res, sig_th=sig_th, sig_arr=sig_arr, symbol="*", type=sig_type, show_ns=True)


def mixture_curve_fit(f, xs, ys, p0, p1, sigma, bounds):
    converged = False
    p1_bounds = bounds[:3] + ((0, 0),)
    while not converged:
        p1_start = p1
        curve_to_fit_p0 = lambda x, *p: f(x, *p1) + f(x, *p)
        p0_res = curve_fit(curve_to_fit_p0, xs, ys, p0=p0, sigma=sigma, bounds=bounds)
        p0 = p0_res[0]
        p0_ste = p0_res[1]
        curve_to_fit_p1 = lambda x, *p: f(x, *p0) + f(x, *p)
        p1_res = curve_fit(curve_to_fit_p1, xs, ys, p0=p1, sigma=sigma, bounds=bounds)
        p1 = p1_res[0]
        p1_ste = p1_res[1]
        error = np.sum((np.array(p1) - np.array(p1_start)) ** 2 )
        if error < 1e-3:
            converged = True
    return [p0, p0_ste], [p1, p1_ste]



def nan_curve_fit(f, xs, g_mean, g_ste, p0, samples=1, ste_compensation=.01, bounds=(-np.inf, np.inf), mixture=False):
    not_nan = ~np.isnan(g_mean)
    g_mean_nan_free = g_mean[not_nan]
    txs_nan_free = xs[not_nan]
    g_ste_nan_free = g_ste[not_nan]+ste_compensation
    if not mixture:
        try:
            g_res = curve_fit(f, txs_nan_free, g_mean_nan_free, p0=p0, sigma=g_ste_nan_free, bounds=bounds)
        except RuntimeError:
            g_res = (p0, np.eye(len(p0)))
        g1_res = None
    else:
        p0, p1 = p0
        g_res, g1_res = mixture_curve_fit(f, txs_nan_free, g_mean_nan_free, p0=p0, p1=p1, sigma=g_ste_nan_free, bounds=bounds)

    g_res_mean = g_res[0]
    g_param_samples = np.random.multivariate_normal(g_res_mean, g_res[1], size=samples)
    if g1_res:
        g1_param_samples = np.random.multivariate_normal(g1_res[0], g1_res[1], size=samples)
    else:
        g1_param_samples = None
    return g_param_samples, g1_param_samples



def trace_curve_fit(f, res, xs, p0,  per_animal=False, mark_f = None, min_f=None, bootstrap=0, axes=[], debug=False, weights=None, trunk=[0, -50], bounds=(-np.inf, np.inf), mixture=False, return_r_sq=False, use_break_point=False):
    curve_res = []
    for sess in res:
        sess_res = []
        for g in sess:
            if axes:
                ax = axes.pop(0)
            else:
                ax = None
            if not per_animal:
                g = [np.vstack(g)]
                txs = xs
            else:
                txs = np.tile(xs, len(g))
            # g = [cell x time]
            g = [x[np.sum(~np.isnan(x), axis=1) > 20] for x in g]
            g_mean = np.hstack([np.nanmean(x, axis=0) for x in g])
            g_count = np.hstack([np.sum(~np.isnan(x), axis=0) for x in g])
            g_ste = np.hstack([np.nanstd(x, axis=0) for x in g]) / np.sqrt(g_count)
            txs = txs[trunk[0]:trunk[1]]
            g_mean = g_mean[trunk[0]:trunk[1]]
            g_ste = g_ste[trunk[0]:trunk[1]]
            # g_mean = arr(time) of means
            break_points = []
            '''
            if use_break_point:
                for i in range(bootstrap):
                    idx = np.arange(g_mean.size)
                    idx_sample = np.random.choice(idx, g_mean.size)
                    BASE_LEN = 100
                    SMOOTH_WINDOW=1
                    Z_TH = 3
                    idx_sample = np.sort(idx_sample)
                    base_sample = g_mean[idx_sample[:BASE_LEN]]
                    base_std = np.std(base_sample)
                    th = base_sample.mean() + base_std * Z_TH
                    g_mean_sample = g_mean[idx_sample]
                    base_smooth = np.convolve(
                            g_mean_sample,
                            np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
                            "same",
                    )
                    res = BASE_LEN + np.nonzero(base_smooth[BASE_LEN:] > th)[0][0]
                    res = txs[res]
                    break_points.append(res)
            '''

            if bootstrap == 0:
                g_param_samples, g1_param_samples = nan_curve_fit(f, txs, g_mean, g_ste, p0,  1000, bounds=bounds, mixture=mixtrue)
            else:
                g_param_samples = []
                g1_param_samples = []
                r_sqs = []
                for i in range(bootstrap):
                    idx = np.arange(g_mean.size)
                    idx_sample = np.random.choice(idx, g_mean.size)


                    g_mean_sample = g_mean[idx_sample]
                    g_ste_sample = g_ste[idx_sample]
                    txs_sample = txs[idx_sample]
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            g_param_sample, g1_param_sample = nan_curve_fit(\
                                    f, txs_sample, g_mean_sample, g_ste_sample, p0, 1, mixture=mixture)
                        except OptimizeWarning:
                            print("curve fit failed")
                    g_param_samples.append(g_param_sample)
                    if use_break_point:
                        BASE_LEN = 50
                        SMOOTH_WINDOW=1
                        Z_TH = 4
                        idx_sample = np.sort(idx_sample)
                        base_sample = g_mean[idx_sample[:BASE_LEN]]
                        base_std = np.std(base_sample)
                        th = base_sample.mean() + base_std * Z_TH
                        g_mean_sample = g_mean[idx_sample]
                        res = BASE_LEN + np.nonzero(f(xs, *g_param_sample[0])[BASE_LEN:] > th)[0][0]
                        res = txs[res]
                        break_points.append(res)

                    if return_r_sq:
                        f_sample = f(txs_sample, *g_param_sample[0])
                        ss_total = np.var(g_mean_sample)
                        ss_residual = np.var(g_mean_sample - f_sample)
                        r_sqs.append(1 - ss_residual / ss_total)
                    if g1_param_sample is not None:
                        g1_param_samples.append(g1_param_sample)

                g_param_samples = np.vstack(g_param_samples)
                if g1_param_samples:
                    g1_param_samples = np.vstack(g1_param_samples)
                    
            g_sim = np.array([f(xs, *params) for params in g_param_samples])

            g_sim_upper_err = np.percentile(g_sim, 97.5, axis=0)
            g_sim_lower_err = np.percentile(g_sim, 2.5, axis=0)
            g_res_mean = np.mean(g_param_samples, axis=0)

            if ax:
                ax.errorbar(txs, g_mean, yerr=g_ste, fmt="k.", alpha=.1)
                ax.fill_between(xs, g_sim_upper_err, g_sim_lower_err, alpha=.1)
                ax.plot(xs, f(xs, *g_res_mean), "r-", lw=1)
                if len(g1_param_samples) > 0:
                    g1_res_mean = np.mean(g1_param_samples, axis=0)
                    ax.plot(xs, f(xs, *g1_res_mean), "b-", lw=1)
                    ax.plot(xs, f(xs, *g_res_mean) + f(xs, *g1_res_mean), "g-", lw=1)
            if min_f or mark_f:
                opt_res = []
                for param in g_param_samples:
                    if min_f:
                        res = minimize(lambda x: min_f(x, *param), param[1]-1.)
                        opt_res.append(res.x)
                        if not res.success:
                            print(res.message)
                    elif mark_f:
                        opt_res.append(mark_f(*param))
                if use_break_point:
                    opt_res = break_points
                opt_upper_err = np.percentile(opt_res, 97.5)
                opt_lower_err = np.percentile(opt_res, 2.5)
                opt_mean = np.mean(opt_res)
                opt_y = f(opt_mean, *g_res_mean)
                if ax:
                    ax.plot([opt_mean], [opt_y], 'k.')
                    ax.plot([opt_lower_err, opt_upper_err], [opt_y, opt_y], 'k-', lw=1)
                if min_f:
                    min_f_sim = np.array([min_f(xs, *params) for params in g_param_samples])
                    min_f_sim_upper_err = np.percentile(min_f_sim, 97.5, axis=0)
                    min_f_sim_lower_err = np.percentile(min_f_sim, 2.5, axis=0)
                    if ax and debug:
                        ax2 = ax.twinx()
                        ax2.fill_between(xs, min_f_sim_upper_err, min_f_sim_lower_err)
                        ax2.plot(xs, min_f(xs, *g_res_mean), 'r-')
            elif return_r_sq:
                opt_res = np.array(r_sqs)
            sess_res.append(opt_res)
        curve_res.append(sess_res)
    return curve_res

logistic_d1 = lambda x, k, x0, v_max, v_min: k*(v_max - v_min)*np.exp(-k*(x - x0))/(1 + np.exp(-k*(x - x0)))**2

logistic_d2 = lambda x, k, x0, v_max, v_min: -(-k**2*(v_max - v_min)*np.exp(-k*(x - x0))/(1 + np.exp(-k*(x - x0)))**2 + 2*k**2*(v_max - v_min)*np.exp(-2*k*(x - x0))/(1 + np.exp(-k*(x - x0)))**3)

# logistic_d2 = lambda x, k, x0, v_max, v_min: -(k**2*(v_max - v_min)*np.exp(-k*(x - x0))/(1 + np.exp(-k*(x - x0)))**2 + 2*k**2*(v_max - v_min)*np.exp(-2*k*(x - x0))/(1 + np.exp(-k*(x - x0)))**3)
logistic_d3 = lambda x, k, x0, v_max, v_min: k**3*(v_max - v_min)*np.exp(-k*(x - x0))/(1 + np.exp(-k*(x - x0)))**2 - 6*k**3*(v_max - v_min)*np.exp(-2*k*(x - x0))/(1 + np.exp(-k*(x - x0)))**3 + 6*k**3*(v_max - v_min)*np.exp(-3*k*(x - x0))/(1 + np.exp(-k*(x - x0)))**4
curve_r = lambda x, k, x0, v_max, v_min: ((1 + logistic_d1(x, k, x0, v_max, v_min)**2)**1.5)/logistic_d2(x, k, x0, v_max, v_min)
curvature = lambda x, k, x0, v_max, v_min: -logistic_d2(x, k, x0, v_max, v_min) / ((1 + logistic_d1(x, k, x0, v_max, v_min)**2)**1.5)
mixture_curve_r = lambda x, k, x0, v_max, v_min, k1, x1, v1: np.abs(((1 + (logistic_d1(x, k, x0, v_max - v1, v_min) + logistic_d1(x, k1, x1, v1, 0. ))**2)**1.5)/(logistic_d2(x, k, x0, v_max, v_min) + logistic_d2(x, k1, x1, v1, 0)))
logit = lambda y, k, x0, v_max, v_min: np.log((v_max - v_min) / (y - v_min) - 1) / (-k) + x0
scaled_logistic = lambda x, k, x0, v_max, v_min: (v_max-v_min) / (1 + np.exp(-k*(x - x0))) + v_min
mixture_scaled_logistic = lambda x, k, x0, v_max, v_min, k1, x1, v1: (v_max-v_min-v1) / (1 + np.exp(-k*(x - x0))) + v1/(1 + np.exp(-k1*(x-x1))) + v_min - v1 / 2

# mark_point_funcs
mid_point = lambda k, x0, v_max, v_min: x0
left_bp = lambda k, x0, v_max, v_min: x0 - k * (v_max - v_min) / 2.

def curve_fit_to_mc_stat(curve_res):
    taus = [(np.array(x).squeeze() + 10)*20 for x in curve_res[0]]
    dtau_idx = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    a = None
    dtau = []
    n = len(taus[0])
    for a,b in dtau_idx:
        try:
            dtau.append(np.random.choice(taus[a], n) \
                      - np.random.choice(taus[b], n))
        except IndexError:
            dtau.append(np.zeros(n))
    selector = [np.ones_like(x) for x in taus]
    return [taus, dtau_idx, dtau, a, selector]


def shrink_axis(ax, ratio, orientation="vertical"):
    b = ax.get_position()
    pb = b.get_points()
    if orientation == "vertical":
        pb[1][1] -= ratio * (pb[1][1] - pb[0][1])
    else:
        pb[1][0] -= ratio * (pb[1][0] - pb[0][0])
    b.set_points(pb)
    ax.set_position(b)
    return ax

def move_axis(ax, coord):
    x, y = coord
    b = ax.get_position()
    pb = b.get_points()
    pb[:,1] += y * (pb[1][1] - pb[0][1])
    pb[:,0] += x * (pb[1][0] - pb[0][0])
    b.set_points(pb)
    ax.set_position(b)
    return ax
# In[23]:


    


# In[17]:


def get_sess_names(a):
    sess = ['Home-pre-train', 'Ctx-pre-shock', 'Ctx-post-shock', 'Home-post-train', 'Ctx-test', 'Home-post-test', 'Ctx-pre-shock-1', 'Ctx-pre-shock-2', 'Ctx-test-5min', 'Ctx-test-3min', 'Ctx-shock', 'Ctx-test-last-5min']
    return [sess[i] for i in a]

def cm(x):
    return "{}cm".format(x)

def inch(x):
    return "{}cm".format(x*2.54)

u = 32

def add_img(fig, img_path, x, y, scale):
    '''
    img_root = et.parse(img_path).getroot()
    img_root.set('x',str(u*x))
    img_root.set('y',str(u*y))
    fig.append(img_root)
    '''        
    g=et.SubElement(fig, 'ns0:g', attrib={
        'transform': "translate({} {})scale({} {})".format(u*x,u*y,scale, scale)
        })
    if img_path.endswith("svg"):
        img = et.parse(img_path).getroot()
    elif img_path.endswith("png"):
        with open(img_path, 'rb') as png_file:
            png_img = png_file.read()
        png_img = base64.b64encode(png_img).decode('ascii')
        img=et.Element('image', attrib={
            'xlink:href': "data:image/png;base64," + png_img})
    g.append(img)

    return fig
    
def get_img(img_path):
    return os.path.abspath(os.path.join(misc_dir, img_path))

def write_fig(fig, fig_path):
    with open(fig_path, "w") as fo:
        fo.write('<?xml version="1.0" standalone="no"?>\n')
        fo.write('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n')
        fo.write('"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n')
        fo.write(et.tostring(fig).decode("utf8"))
        fo.write("\n\n")
# figure organization

def draw_stats(all_data, ax, transpose=False, base=.5):
    ax.set_yticks([.5, .6, .7, .8, .9, 1.])
    ps = [ttest_1samp(g, base)[1] for g in all_data]
    print(" ".join(["{:.3f}".format(p) for p in ps]))
    l = len(all_data)
    arg_order = sorted(range(l), key=lambda x: ps[x])
    order = [arg_order.index(i) for i in range(l)]
    for i, g, o in zip(range(l), all_data, order):
        margin = .05
        h = np.mean(g) + np.std(g)/np.sqrt(len(g)) + margin
        if np.mean(g) < base:
            h = base + margin
        if transpose:
            ti = i % 2 * 4 + i // 2
            ix = ti + ti // 4 * .5 
        else:
            ix = i + i // 3 * .5
        x,y = d2ax(ax, np.array([(ix,h)]))[0]
        tres = ttest_1samp(g, base)
        _,p = tres
        p *= l - o
        symbol = "n.s."
        if p < .05:
            symbol = "*"
        if p < .01:
            symbol = "**"
        if p < .001:
            symbol = "***"
        ax.text(x, y, symbol, fontsize = fontsize, horizontalalignment="center", verticalalignment="center", clip_on=False, transform=ax.transAxes)
