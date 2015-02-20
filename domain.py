import numpy
from PIL import Image
import colorsys
import argparse
import re as regex
import time

# h <- (arg(z)/2pi+1) % 1
# s <- [0, 1]
# v <- g(|z|)

def compl_h(z):
    h = numpy.angle(z) / (2 * numpy.pi) + 1
    return numpy.mod(h, 1)


def rlen(region, nodes):
    return (max(region) - min(region)) * nodes


def compl_grid(func, x, y, n):
    width = max(x) - min(x)
    height = max(y) - min(y)
    x = numpy.linspace(min(x), max(x), float(width) * n)
    y = numpy.linspace(min(y), max(y), float(height) * n)
    x, y = numpy.meshgrid(x, y)
    zgrid = x + 1j * y
    wgrid = func(zgrid)
    return numpy.flipud(wgrid)


def plot_grid(col, f, imgname, re=[-1.0,1.0], im=[-1.0,1.0], nodes=200, s=0.9):
    grid = compl_grid(f, re, im, nodes)
    coloured = col(grid, s)
    width = int(rlen(re, nodes))
    height = int(rlen(im, nodes))
    img = Image.new("RGB", (width, height), "white")
    pix = img.load()
    for y, j in enumerate(coloured):
        for x, i in enumerate(j):
            rgb = colorsys.hsv_to_rgb(i[0], i[1], i[2])
            pix[x, y] = tuple([int(n * 255) for n in rgb])
    img.save(imgname)
    

def dc_classic(grid, sfac):
    indi = numpy.where(numpy.isinf(grid))
    indn = numpy.where(numpy.isnan(grid))
    h = compl_h(grid)
    s = sfac * numpy.ones_like(h)
    mod = numpy.absolute(grid)
    v = (1.0 - 1.0/(1+mod ** 2))**0.2
    h[indi] = 0.0
    s[indi] = 0.0
    v[indi] = 1.0
    h[indn] = 0
    s[indn] = 0
    v[indn] = 0.5
    hsv = numpy.dstack((h,s,v))
    return hsv


def dc_modtrack(grid, sfac):
    h = compl_h(grid)
    mod = numpy.absolute(grid)
    c = numpy.log(2)
    logm = numpy.log(mod)
    logm = numpy.nan_to_num(logm)
    v = logm - numpy.floor(logm)
    s = sfac * numpy.ones_like(h, float)
    hsv = numpy.dstack((h,s,v**0.2))
    return hsv


def function_parser(func):
    ret = lambda z: z
    if func:
        blocks = regex.findall(r"([\/\-\+])?([0-9]+[\.]?[0-9]*?i?)?([\(])?(([\-\+])?([0-9]+[\.]?[0-9]*?i?)?[z|0-9][\.]?[0-9]*?i?([\^][0-9]+[\.]?[0-9]*?i?)?)([/)])?", func)
        if blocks:
            blocks = [''.join(b[:6]+b[7:]) for b in blocks]
            fstr = [regex.sub(r"([1-9]+)(z)", r"\1*\2", n) for n in blocks]
            fstr = [regex.sub(r"\^", "**", n) for n in fstr]
            fstr = [s.replace("i", "j") for s in fstr]
            ret = lambda z: eval(''.join(fstr))
    return ret


def colour_arg_parser(col):
    func = dc_modtrack
    if col == "classic":
        func = dc_classic
    return func
############
parser = argparse.ArgumentParser(description='Plot some complex numbers!')
parser.add_argument('filename', help="Specify the name of the file to write output to.")
parser.add_argument('-f', '--func', help="""Use a custom function to plot. Use the letter 'z' as a variable for polynomials.\n
                                            Some terminals may require \"\" marks around more complex functions (i.e. those with
                                            parentheses or other special characters). They are not necessary for all functions
                                            but are recommended.""")
parser.add_argument('-n', '--nodes', help="Specify the number of nodes in the plotted region (similar to resolution).",
                    type=int)
parser.add_argument('-c', '--colour', help="Choose a domain colouring method",
                    default="modulus", choices=["classic", "modulus"])
parser.add_argument('-d', '--debug', help="Debug mode.", action="store_true")
parser.add_argument('-t', '--time', help="Display the execution time of the program.", action="store_true")
parser.add_argument('--realmin', help="Minimum value to display on the real axis", type=float, default=-1.0)
parser.add_argument('--realmax', help="Maximum value to display on the real axis", type=float, default=1.0)
parser.add_argument('--imagmin', help="Minimum value to display on the imaginary axis", type=float, default=-1.0)
parser.add_argument('--imagmax', help="Maximum value to display on the imaginary axis", type=float, default=1.0)
args = parser.parse_args()
if args.nodes:
    nodes = args.nodes
else:
    nodes = 200

re = [-1.0, 1.0]
im = [-1.0, 1.0]
if args.realmin:
    re[0] = args.realmin
if args.realmax:
    re[1] = args.realmax
if args.imagmin:
    im[0] = args.imagmin
if args.imagmax:
    im[1] = args.imagmax

cf = colour_arg_parser(args.colour)
f = function_parser(args.func)

if not args.debug:
    print "Starting generation..."
    start = time.time()
    plot_grid(cf, f, args.filename, re, im, nodes)
    end = time.time()
    print "Generation ended."
    print "Filename:\t", args.filename
    print "Region:\t\t", "Real[", re[0], " -> ", re[1], "]"
    print "\t\t", "Imag[", im[0], " -> ", im[1], "]"
    print "Function:\t",
    if args.func:
        print "f(z) =", args.func
    else:
        print "f(z) = z"
    if args.time:
        print "Time:\t\t", end - start, "seconds."
else:
    print "Debugging"
