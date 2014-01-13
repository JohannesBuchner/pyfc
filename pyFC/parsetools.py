"""
________________________________________________________________
Description

Here is some code to assist in parsing text. 
Currently only a unique filename creatiion routine.

"""


import os.path as osp
import glob
import re

def unique_fname(fname, sep='-', upat='[0-9][0-9]'):
    """
    Create unique filename, continuing a filename sequence if one exists
    already, with unique pattern <upat> and separateor <sep>. <fname> is logically
    thought of as consisting of <base><sep><upat><ext>. At the least, it equates
    to <base>. <base> and <ext> are inferred from <fname>.

    A filename sequence is defined by:  <base><sep><upat><ext>
    and for the default would be     :  file-[0-9][0-9].dat

    This function returns:
        filename with <upat> incremented, if files matching the pattern exist; or
        filename with first <upate> appended, if 

    Current caveats and todos: 
        - Works only for default (numeral) <upat> currently, because number of
          digits must be 2. This is easy to generalize, though.
        - More error checking for when <upat>
    """
    ext = osp.splitext(fname)[1]
    base = osp.splitext(fname)[0]

    if re.search(sep+upat+'$', base):
        num = base.rsplit(sep, 1)[1]
        base = base.rsplit(sep, 1)[0]

    files = sorted(glob.glob(base+'-'+upat+ext))
    if files:
        num = osp.splitext(files[-1])[0].rsplit(sep, 1)[1]
        fname = base + sep + format(int(num)+1, '02d') + ext

    elif glob.glob(fname): fname = base + '-01' + ext

    return fname

