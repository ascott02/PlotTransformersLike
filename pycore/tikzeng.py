
import os, sys

# Compatibility layer additions for transformer-style diagrams.
# These helpers mimic the original string-based API without depending
# on the new Python object model so legacy scripts can be updated with
# minimal changes.

def _tf_node(style, name, x, y, w, h, label):
    return f"\\node[{style}={w:.2f}cm/{h:.2f}cm] ({name}) at ({x:.2f}cm,{y:.2f}cm) {{{label}}};\n"

def to_LayerNorm(name, x, y, w=2.6, h=0.6, text="LayerNorm"):
    return _tf_node("sblk", name, x, y, w, h, text)

def to_MHA(name, x, y, w=3.8, h=1.2, heads=8, d_model=768, masked=False):
    star = "*" if masked else ""
    label = f"MHA{star}\\\\\\scriptsize(h={heads}, $d_{{model}}={d_model}$)"
    return _tf_node("blk", name, x, y, w, h, label)

def to_FFN(name, x, y, w=3.2, h=1.2, dff=3072):
    label = f"FFN\\\\\\scriptsize($d_{{ff}}={dff}$)"
    return _tf_node("blk", name, x, y, w, h, label)

def to_Add(name, x, y, r=0.22):  # add node expects same w/h param twice
    return f"\\node[addnode={r:.2f}cm/{r:.2f}cm] ({name}) at ({x:.2f}cm,{y:.2f}cm) {{$+$}};\n"

def to_CLSHead(name, x, y, w=2.4, h=1.0, classes=1000):
    label = f"CLS Head\\\\\\scriptsize($C={classes}$)"
    return _tf_node("blk", name, x, y, w, h, label)

def to_transformer_block(name_prefix, x, y):
    """Return snippets for a minimal encoder-style block (LN -> MHA -> LN -> FFN -> Add)."""
    snippets = []
    snippets.append(to_LayerNorm(f"{name_prefix}ln1", x, y))
    snippets.append(to_MHA(f"{name_prefix}mha", x + 3.2, y))
    snippets.append(to_LayerNorm(f"{name_prefix}ln2", x + 7.4, y))
    snippets.append(to_FFN(f"{name_prefix}ffn", x + 10.6, y))
    snippets.append(to_Add(f"{name_prefix}add", x + 14.4, y + 0.2))
    return snippets

def to_xt_transformer_block(name_prefix, x, y):  # backwards alias
    return to_transformer_block(name_prefix, x, y)

def to_decoder_block(name_prefix, x, y, include_cross=True):  # pragma: no cover
    """Return snippets for a simplified decoder block.

    Layout: LN -> MHA* -> Add -> LN -> (CrossAttn -> Add)? -> LN -> FFN -> Add
    The spacing mirrors the object API for visual consistency.
    """
    cur = x
    snips = []
    snips.append(to_LayerNorm(f"{name_prefix}ln1", cur, y))
    snips.append(to_MHA(f"{name_prefix}mha", cur + 3.2, y, masked=True))
    snips.append(to_Add(f"{name_prefix}add1", cur + 5.6, y + 0.2))
    cur += 7.3
    if include_cross:
        snips.append(to_LayerNorm(f"{name_prefix}ln2", cur, y))
        snips.append(to_MHA(f"{name_prefix}xatt", cur + 3.2, y))
        snips.append(to_Add(f"{name_prefix}add2", cur + 5.6, y + 0.2))
        cur += 7.3
    snips.append(to_LayerNorm(f"{name_prefix}ln3", cur, y))
    snips.append(to_FFN(f"{name_prefix}ffn", cur + 3.0, y))
    snips.append(to_Add(f"{name_prefix}add3", cur + 5.4, y + 0.2))
    return snips

def to_head( projectpath ):
    pathlayers = os.path.join( projectpath, 'layers/' ).replace('\\', '/')
    return r"""
\documentclass[border=8pt, multi, tikz]{standalone} 
\usepackage{import}
\subimport{"""+ pathlayers + r"""}{init}
% Extended transformer styles
\input{transformer_tex/transformer_styles.tex}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image 
"""

def to_cor():
    return r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
\def\SumColor{rgb:blue,5;green,15}
"""

def to_begin():
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""

# layers definition

def to_input( pathfile, to='(-3,0,0)', width=8, height=8, name="temp" ):
    return r"""
\node[canvas is zy plane at x=0] (""" + name + """) at """+ to +""" {\includegraphics[width="""+ str(width)+"cm"+""",height="""+ str(height)+"cm"+"""]{"""+ pathfile +"""}};
"""

# Conv
def to_Conv( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +""",
        fill=\ConvColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# Conv,Conv,relu
# Bottleneck
def to_ConvConvRelu( name, s_filer=256, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {RightBandedBox={
        name="""+ name +""",
        caption="""+ caption +""",
        xlabel={{ """+ str(n_filer[0]) +""", """+ str(n_filer[1]) +""" }},
        zlabel="""+ str(s_filer) +""",
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +""",
        width={ """+ str(width[0]) +""" , """+ str(width[1]) +""" },
        depth="""+ str(depth) +"""
        }
    };
"""



# Pool
def to_Pool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+name+""",
        caption="""+ caption +r""",
        fill=\PoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# unpool4, 
def to_UnPool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+ name +r""",
        caption="""+ caption +r""",
        fill=\UnpoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""



def to_ConvRes( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=6, height=40, depth=40, opacity=0.2, caption=" " ):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {RightBandedBox={
        name="""+ name + """,
        caption="""+ caption + """,
        xlabel={{ """+ str(n_filer) + """, }},
        zlabel="""+ str(s_filer) +r""",
        fill={rgb:white,1;black,3},
        bandfill={rgb:white,1;black,2},
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""


# ConvSoftMax
def to_ConvSoftMax( name, s_filer=40, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# SoftMax
def to_SoftMax( name, s_filer=10, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        xlabel={{" ","dummy"}},
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Sum( name, offset="(0,0,0)", to="(0,0,0)", radius=2.5, opacity=0.6):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Ball={
        name=""" + name +""",
        fill=\SumColor,
        opacity="""+ str(opacity) +""",
        radius="""+ str(radius) +""",
        logo=$+$
        }
    };
"""


def to_connection( of, to):
    return r"""
\draw [connection]  ("""+of+"""-east)    -- node {\midarrow} ("""+to+"""-west);
"""

def to_skip( of, to, pos=1.25):
    return r"""
\path ("""+ of +"""-southeast) -- ("""+ of +"""-northeast) coordinate[pos="""+ str(pos) +"""] ("""+ of +"""-top) ;
\path ("""+ to +"""-south)  -- ("""+ to +"""-north)  coordinate[pos="""+ str(pos) +"""] ("""+ to +"""-top) ;
\draw [copyconnection]  ("""+of+"""-northeast)  
-- node {\copymidarrow}("""+of+"""-top)
-- node {\copymidarrow}("""+to+"""-top)
-- node {\copymidarrow} ("""+to+"""-north);
"""

def to_end():
    return r"""
\end{tikzpicture}
\end{document}
"""


def to_generate( arch, pathname="file.tex" ):
    with open(pathname, "w") as f: 
        for c in arch:
            print(c)
            f.write( c )
     


