import numpy as np
import pandas as pd


"""
Dokumentasi
- teori: link/hubungan antara konsep
- konsep: objek yang terlibat dalam suatu teori
- contoh run
"""

def flattenimg(img):
    # reshape to 2-D
    if len(img.shape) > 2 and img.shape[-1] > 2:
        return img.reshape(-1, img.shape[-1])
    else:
        return img.reshape(img.shape[0]*img.shape[1])

def sortimg(img):
    # obtain indices of a sorted matrix
    id = np.lexsort(img.T)
    # rearrange matrix based on this index and return
    return  img[id, :]

""" Returns separate histogram for each color in a pixel: 3 for RGB, 4 for CMYK, 1 for Grayscale
"""
def gethistogram(img):
    # reshape image
    imgr = flattenimg(img)
    # get histogram for each color the numpy way: use this!
    hist = []
    [hist.append(np.histogram(imgr[:,n], bins=256)[0]) for n in range(imgr.shape[-1])]

    return np.asarray(hist)

"""
    # horribly slow vanilla implementation: use only for proof of work
    hist = np.zeros((256, imgr.shape[-1]), dtype=np.uint32)
    for pixel in imgr:
        for color in range(imgr.shape[-1]):
            hist[pixel.item(color), color] += 1
    return hist
"""

def equalize(img):
    # generate lookup table(s)
    imgsize = img.shape[0]*img.shape[1]
    lut = []

    # generate histogram
    hist = gethistogram(img)

    # if grayscale add an axis
    gs = False
    if len(img.shape) < 3:
        gs = True
        img = img.reshape((img.shape[0], img.shape[1], 1))

    # for each color (1 for greyscale, 3 for RGB, 4 for CMYK)
    for color in range(img.shape[-1]):
        # generate cumulative distribution function
        cdf = hist[color].cumsum()
        # retrieve first non-zero element of cdf
        cdfmin = 0
        while cdf.item(cdfmin) == 0:
            cdfmin += 1
        # equalize the colors / generate lookup table
        norm = (cdf - cdfmin)*255/(imgsize - cdfmin)
        lut.append(norm)

    lut = np.asarray(lut, dtype=np.uint8)

    # remap colors in img according to lut (lookup table):
    eq = np.dstack([lut[n][img[...,n]] for n in range(img.shape[-1])])
    # if grayscale remove added axis
    if gs:
        eq = eq.reshape((img.shape[0], img.shape[1]))
    return eq

def getgrayscale(img):
    return np.asarray((0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]), dtype=np.uint8)

def getunique(imgs):
    # detect unique colors:
    # diff() along image length to detect changes in pixel color
    # any() along axis 0 to count number of color changes
    return np.any(np.diff(imgs, axis=0), axis=1).sum() + 1

def getbackground(img, imgs, thrs=10):
    # count occurences of each unique pixel color
    id = np.append([0], np.any(np.diff(imgs, axis=0), axis=-1).cumsum())
    count = np.bincount(id)
    # calculate index of most common pixel
    idc = 0
    if np.argmax(count) > 0:
        idc = count[:np.argmax(count)].cumsum()[-1]
    # obtain most common pixel
    pxc = imgs[idc]
    # for each pixel in original image obtain its difference to pxc.
    # if the absolute of the difference between ANY color in the two
    # is below a certain threshold return TRUE for that pixel, hence
    # indentify it as a background pixel.
    # else return FALSE (foreground).
    back = np.any(np.abs(img - pxc) <= thrs, axis=-1)

    return back

def degreezero(source, type="average"):
    if len(source.shape) == 3:
        imgb = getgrayscale(source)
    else: imgb = np.copy(source)

    imgt = np.zeros((imgb.shape), dtype=imgb.dtype)
    sub = np.zeros((3,3), dtype=int)

    if type == "average":
        for row in range(1, imgb.shape[0] - 1):
            for col in range(1, imgb.shape[1] - 1):
                sub[:] = imgb[row-1:row+2, col-1:col+2]
                imgt.itemset((row,col), np.sum(sub)/9.)

    elif type == "difference":
        for row in range(1, imgb.shape[0] - 1):
            for col in range(1, imgb.shape[1] - 1):
                sub[:] = imgb[row-1:row+2, col-1:col+2]
                imgt.itemset((row,col), np.abs([sub[0,0] - sub[2,2], sub[0,1] - sub[2,1], sub[0,2] - sub[2,0], sub[1,2] - sub[1,0]]).max())

    elif type == "homogen":
        for row in range(1, imgb.shape[0] - 1):
            for col in range(1, imgb.shape[1] - 1):
                sub[:] = imgb[row-1:row+2, col-1:col+2]
                sub[:] -= sub[1,1]
                imgt.itemset((row,col), np.abs(sub).max())

    return imgt

def convolve(filt, source):
    if len(source.shape) == 3:
        imgb = getgrayscale(source)
    else: imgb = np.copy(source)

    imgt = np.zeros((imgb.shape))
    sub = np.zeros((3,3), dtype=imgb.dtype)

    for row in range(1, imgb.shape[0] - 1):
        for col in range(1, imgb.shape[1] - 1):
            sub[:] = imgb[row-1:row+2, col-1:col+2]
            imgt.itemset((row,col), np.sum(sub * filt)/9.)

    return imgt

"""
Make sure that source image is grayscale before using this function
"""
def convolvefft(filt, source):
    return np.fft.irfft2(np.fft.rfft2(source) * np.fft.rfft2(filt, source.shape))

def degreeone(source, type="sobel", fft=True):
    if len(source.shape) == 3:
        imgb = getgrayscale(source)
    else: imgb = np.copy(source)

    imgt = np.zeros((imgb.shape), dtype=np.uint8)

    if type == "sobel":
        horizontal = np.array(([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), dtype=float)
        vertical = np.array(([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), dtype=float)
        if fft:
            imgt = np.sqrt(np.square(convolvefft(horizontal, imgb)) + np.square(convolvefft(vertical, imgb))).astype(np.uint8)
        else:
            imgt = np.abs(convolve(horizontal, imgb) + convolve(vertical, imgb)).astype(np.uint8)

    elif type == "prewitt":
        horizontal = np.array(([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), dtype=float)
        vertical = np.array(([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), dtype=float)
        if fft:
            imgt = np.sqrt(np.square(convolvefft(horizontal, imgb)) + np.square(convolvefft(vertical, imgb))).astype(np.uint8)
        else:
            imgt = np.abs(convolve(horizontal, imgb) + convolve(vertical, imgb)).astype(np.uint8)

    elif type == "freichen":
        sq2 = 2**0.5
        g1 = (0.5/sq2)*np.array([[1, sq2, 1], [0, 0, 0], [-1, -sq2, -1]], dtype=float)
        g2 = (0.5/sq2)*np.array([[1, 0, -1], [sq2, 0, -sq2], [1, 0, -1]], dtype=float)
        g3 = (0.5/sq2)*np.array([[0, -1, sq2], [1, 0, -1], [-sq2, 1, 0]], dtype=float)
        g4 = (0.5/sq2)*np.array([[sq2, -1, 0], [-1, 0, 1], [0, 1, -sq2]], dtype=float)
        g5 = 0.5*np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]], dtype=float)
        g6 = 0.5*np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=float)
        g7 = (1./6)*np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
        g8 = (1./6)*np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]], dtype=float)
        g9 = (1./3)*np.ones((3,3), dtype=float)

        if fft:
            mask1 = np.sqrt(np.square(convolvefft(g1, imgb)) + np.square(convolvefft(g2, imgb)) + np.square(convolvefft(g3, imgb)) + np.square(convolvefft(g4, imgb)))
            imgt = mask1.astype(np.uint8)

    return imgt

def degreetwo(source, type="kirsch"):
    if len(source.shape) == 3:
        imgb = getgrayscale(source)
    else:
        imgb = np.copy(source)

    filt = np.zeros((8), dtype=int)
    if type == "kirsch":
        filt[:] = np.array([5, 5, 5, -3, -3, -3, -3, -3])
    elif type == "prewitt":
        filt[:] = np.array([1, 1, -1, -1, -1, 1, 1, 1])

    temp = []

    for n in range(8):
        if type == "kirsch":
            rollfilt = np.insert(np.roll(filt, n), 4, 0).reshape((3,3))
        elif type == "prewitt":
            rollfilt = np.insert(np.roll(filt, n), 4, -2).reshape((3,3))
        temp.append(convolvefft(rollfilt, imgb))

    temp = np.dstack(temp)
    imgt = np.max(temp, axis=-1)
    imgt = (255*imgt/imgt.max()).astype(np.uint8)

    return imgt

def gaussian_filt(shape=(5,5), sigma=0.8):
    m, n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[m-1:m+2, n-1:n+2]

    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[h <  np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h

def downsample(img, target_height=320):
    if img.shape[0] < target_height/2:
        return img
    else:
        import scipy.ndimage as scimg
        ratio = target_height*1./img.shape[0]
        return scimg.zoom(img, (ratio, ratio, 1), order=0)

    return

"""
Returns the color range of an input skin image.
The colorrange variable determines the maximum range; if the color of the input image is
wider than the colorrange then the range such that the maximum number of pixels falls 
within that range is used.
"""
def mapColor(skin, colorrange=60):
    r = skin[...,0]
    g = skin[...,1]
    b = skin[...,2]

    hist_r = np.histogram(r, bins=255, range=(0,255))[0]
    hist_g = np.histogram(g, bins=255, range=(0,255))[0]
    hist_b = np.histogram(b, bins=255, range=(0,255))[0]

    if r.ptp() > colorrange:
        max_r = []
        [max_r.append(np.sum(hist_r[n:n+colorrange])) for n in range(r.min(), r.max()-colorrange)]
        range_r = np.argmax(max_r) + r.min()
        range_r = [range_r, range_r+colorrange]
    else:
        range_r = [r.min(), r.max()]

    if g.ptp() > colorrange:
        max_g = []
        [max_g.append(np.sum(hist_g[n:n+colorrange])) for n in range(g.min(), g.max()-colorrange)]
        range_g = np.argmax(max_g) + g.min()
        range_g = [range_g, range_g+colorrange]
    else:
        range_g = [g.min(), g.max()]

    if b.ptp() > colorrange:
        max_b = []
        [max_b.append(np.sum(hist_b[n:n+colorrange])) for n in range(b.min(), b.max()-colorrange)]
        range_b = np.argmax(max_b) + b.min()
        range_b = [range_b, range_b+colorrange]
    else:
        range_b = [b.min(), b.max()]

    return np.array([range_r, range_g, range_b]).T

"""
Returns a binarized version of the input image such that the pixels corresponding to the
colormap are True and the rest are False.
"""
def mapImage(img, colormap):
    return np.logical_and(np.all(img > colormap[0], axis=-1), np.all(img < colormap[1], axis=-1))

"""
Returns an image such that the pixels in objlist are surrounded by a green box.
"""
def boxobj(img, objlist):
    imgf = np.copy(img)
    for obj in objlist:
        objnp = np.array(obj)
        y = objnp[:,0]
        x = objnp[:,1]
        y0 = y.min()
        y1 = y.max()
        x0 = x.min()
        x1 = x.max()
        green = [0,  255, 0]
        imgf[y0, x0:x1] = green
        imgf[y1, x0:x1] = green
        imgf[y0:y1, x0] = green
        imgf[y0:y1, x1] = green

    return imgf

def colorobj(img, objlist):
    imgf = np.copy(img)

    objs = np.vstack(objlist)
    objsY = objs[:,0]
    objsX = objs[:,1]

    red = [255, 0, 0]

    imgf = np.array(imgf*0.25, dtype=np.uint8)
    imgf[objsY, objsX] = red
    
    return imgf

def showobj(img, objlist, color=True, box=True, opac=1):
    if len(img.shape) > 2:
        imgf = np.array(np.copy(img)*opac, dtype=np.uint8)

    green = [0,  255, 0]
    startcolor = np.array([0, 128, 255], dtype=np.uint8)
    if objlist:
        incr = 255/len(objlist)
    else:
        incr = 1

    for obj in objlist:
        objnp = np.array(obj)
        y = objnp[:,0]
        x = objnp[:,1]

        if box:
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            imgf[y0, x0:x1] = green
            imgf[y1, x0:x1] = green
            imgf[y0:y1, x0] = green
            imgf[y0:y1, x1] = green

        if color:
            startcolor += incr
            imgf[y, x] = startcolor

    return imgf


"""
Returns an image such that the pixels in objlist are displayed whereas the rest of the
pixels are less transparent.
"""
def processFaces(img, objlist, transparency=0, minvar=500):
    if transparency == 0:
        mask = np.zeros(img.shape, dtype=np.uint8)
    else:
        mask = np.array(img*transparency, dtype=np.uint8)

    for f in objlist:
        face = np.array(f)
        y = face[...,0]
        x = face[...,1]
        # hardwired height:width ratio
        height = y.ptp()
        width = x.ptp()
        #if width*1./height > 1.2: continue
        #if y.size*1./y.ptp() < 1.: continue
        pix = img[y.min():y.max(), x.min():x.max()]
        vari = np.var(np.sum(pix, axis=-1)/3)
        if vari < minvar: continue
        if face.size*1./(height*width) < 0.3: continue
        mask[y.min():y.max(), x.min():x.max()] = img[y.min():y.max(), x.min():x.max()]

    return mask

def processFace(img, objlist, transparency=0, usemask=True):
    if transparency == 0:
        mask = np.zeros(img.shape, dtype=np.uint8)
    else:
        mask = np.empty(img*transparency, dtype=np.uint8)

    sizes = []

    for f in objlist:
        obj = np.array(f)
        y = obj[...,0]
        x = obj[...,1]

        sizes.append(y.ptp()*x.ptp())

    face = np.array(objlist[np.array(sizes).argmax()])

    y = face[...,0]
    x = face[...,1]
    # hardwired height:width ratio
    #height = y.ptp()
    #width = x.ptp()
    #if width*1./height > 1.2: continue
    #if y.size*1./y.ptp() < 1.: continue
    if usemask:
        mask[y.min():y.max(), x.min():x.max()] = img[y.min():y.max(), x.min():x.max()]
    else:
        mask = img[y.min():y.max(), x.min():x.max()]

    return mask

def cleanFaces(img, objlist, transparency=0):
    if transparency == 0:
        mask = np.zeros(img.shape, dtype=np.uint8)
    else:
        mask = np.array(img*transparency, dtype=np.uint8)

    for f in objlist:
        face = np.array(f)
        y = face[...,0]
        x = face[...,1]
        # hardwired height:width ratio
        height = y.ptp()
        width = x.ptp()
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        #if width*1./height > 1.2: continue
        #if y.size*1./y.ptp() < 1.: continue
        mask[y.min():y.max(), x.min():x.max()] = img[y.min():y.max(), x.min():x.max()]

    return mask

def getFaces(img, skin, range=60):
    if np.any(np.array(img.shape) > 1200):
        imgr = downsample(img)
    else:
        imgr = img

    facesbin = mapImage(imgr, mapColor(skin, range))
    objlist = segment(thin(facesbin))
    #faces = processFaces(imgr, objlist)
    faceproc = processFaces(imgr, objlist, transparency=0.25)
    #objlist2 = segment(mapImage(faceproc, mapColor(skins, 100)))
    #faceproc2 = processFaces(imgr, objlist2, transparency=0.25)
    #faces = boxobj(faceproc2, objlist2)

    return faceproc

def getFaceFeats(img, skin, range=100):
    if np.any(np.array(img.shape) > 1200):
        imgr = downsample(img)
    else:
        imgr = img

    # Get the object which is most likely the outline of the face
    # Choose the longest object
    facebin = mapImage(imgr, mapColor(skin, range))
    faceobj = segment(thin(facebin))
    facesize = []
    for obj in faceobj:
        objnp = np.array(obj)
        ptpY = objnp[:,0].ptp()
        ptpX = objnp[:,1].ptp()
        facesize.append(ptpY*ptpX)

    faceout = np.array(faceobj[np.argmax(np.array(facesize))])

    # Get the objects which contain, among others, the facial features
    """
    otsu1 = otsu(imgr)
    otsu2 = otsu(otsu1*getgrayscale(imgr))
    featobj = segment(thin(otsu1*otsu2))
    drawobj(imgr, np.vstack(featobj), opac=0.1)
    """

    gauss = gaussian_filt(shape=(10, 10), sigma=1.2)
    faceedge = degreeone(imgr)
    faceblur = convolvefft(gauss, faceedge)
    facein = otsu(faceblur, bg='light')
    
    featobj = segment(thin(facein))

    # Determine which objects in featobj are inside faceout
    return objsinobj(faceout, featobj)

"""
Accepts a reference object described by a list of pixel coordinates and a list of objects described by lists of pixel coordinates. Return a list of objects inside the reference object.
"""
def objsinobj(refobj, objlist):
    inobj = []

    faceY = refobj[..., 0]
    faceX = refobj[..., 1]

    for obj in objlist:
        inside = True
        objnp = np.array(obj)
        objY = objnp[..., 0]
        objX = objnp[..., 1]

        if objY.min() < faceY.min() or objY.max() > faceY.max() or objX.min() < faceX.min() or objX.max() > faceX.max():
            inside = False
        else:
            for pixel in objnp:
                faceYwhere = np.argwhere(faceY == pixel[0])
                faceXwhere = np.argwhere(faceX == pixel[1])
                if faceYwhere.size < 2 or faceXwhere.size < 2:
                    inside = False
                    break
        if inside:
            inobj += [objnp]
                
    return inobj

"""
turn into grayscale, equalize, threshold
"""
def binarize(img, bg='dark'):
    # grayscale conversion
    gs = np.asarray((0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]), dtype=np.uint8)

    # gaussian filter
    gauss = gaussian_filt()
    pass1 = convolvefft(gauss, gs)

    # histogram and cumulative sum
    hist = np.histogram(pass1, bins=256)[0]
    cdf = hist.cumsum()

    """
    # equalize
    cdfmin = 0
    while cdf.item(cdfmin) == 0: cdfmin += 1
    lut = (cdf - cdfmin)*255/(gs.size - cdfmin)
    eq = np.uint8(lut[pass1])
    """

    # gaussian filter
    pass2 = convolvefft(gauss, pass1)

    # histogram and cumulative sum
    hist = np.histogram(pass2, bins=256)[0]
    cdf = hist.cumsum()

    # otsu threshold
    sumlist = hist*np.arange(256)
    sumtot = np.sum(sumlist)
    sumcum = sumlist.cumsum()
    tot = pass2.size

    wf = tot - cdf
    mb = sumcum/cdf
    mf = (sumtot - sumcum)/wf
    thr = cdf*wf*np.square(mb-mf)

    """
    for n in range(cdfmin, 256):
        wb = cdf.item(n)
        wf = tot - wb
        if wf == 0: break
        sumb = sumcum.item(n)
        mb = 1.0*sumb/wb
        mf = 1.0*(sumtot - sumb)/wf
        thr.append(wb*wf*(mb-mf)**2)
    """
    if bg == 'dark':
        return pass2 > np.argmax(thr) + 1
    else:
        return pass2 < np.argmax(thr) + 1

"""
    applies otsu's automatic thresholding algorithm to separate
    background and foreground:
    1. Compute histogram and probabilities of intensity levels
    2. Initialize wb and mb
    3. Step through each intensity level and compute:
    4. wb and mb for that level
    5. var**2b for that level
    6. Once found for all levels, take maximum var**2
"""
def otsu(img, bg='dark'):
    # compute grayscale version of image
    if len(img.shape) == 3:
        imgg = getgrayscale(img)
    else: imgg = np.copy(img)

    # compute histogram and probabilities
    hist = np.histogram(imgg, bins=256)[0]
    cdf = hist.cumsum()

    import matplotlib.pyplot as plt

    # otsu threshold
    sumlist = hist*np.arange(256)
    sumtot = np.sum(sumlist)
    sumcum = sumlist.cumsum()
    tot = imgg.size

    wf = tot - cdf
    mb = sumcum/cdf
    mf = (sumtot - sumcum)/wf
    thr = (cdf*wf*np.square(mb-mf))[:-1]

    lvl = np.argmax(thr) + 1
    if bg == 'dark':
        return(imgg <= lvl)
    else:
        return(imgg > lvl)

def otsurgb(img, bg='dark'):
    return np.dstack((otsu(img[...,0], bg), otsu(img[...,1], bg), otsu(img[...,2], bg)))

def zhangsuen(img):
    # copy of original (binary) image
    imgt = np.copy(img)
    obj = np.argwhere(img)
    # list of pixel coordinates
    # account for border cases by removing pixels on border from list
    list = np.array([o for o in obj if o[0] > 1 and o[0] < imgt.shape[0]-1 and o[1] > 1 and o[1] < imgt.shape[1]-1])

    # initialize dummy value for mark_for_deletion list to get while loop going
    mark_for_deletion = [0]

    while mark_for_deletion:
        mark_for_deletion = []

        """
        first pass
        """
        for npix in range(list.shape[0]):
            # array of circling neighbours
            pix = list[npix,:]
            sub = np.array([imgt.item(pix[0]-1, pix[1]-1), imgt.item(pix[0]-1, pix[1]), imgt.item(pix[0]-1, pix[1]+1), imgt.item(pix[0], pix[1]+1), imgt.item(pix[0]+1, pix[1]+1), imgt.item(pix[0]+1, pix[1]), imgt.item(pix[0]+1, pix[1]-1), imgt.item(pix[0], pix[1]-1)])

            # if a pixel satisfies these conditions then mark it for deletion
            if (2 <= np.sum(sub) <= 6 and
                np.sum(np.diff(sub)) <= 2 and 
                not (sub.item(1) and sub.item(3) and sub.item(5)) and 
                not (sub.item(3) and sub.item(5) and sub.item(7))):
                mark_for_deletion.append(npix)

        # erase marked pixels from image
        [imgt.itemset((list[n,0], list[n,1]), False) for n in mark_for_deletion]
        # erase marked pixels from list of non-zero pixels
        list = np.delete(list, mark_for_deletion, axis=0)
        # begin anew
        mark_for_deletion = []

        """
        second pass
        """
        for npix in range(list.shape[0]):
            pix = list[npix,:]
            sub = np.array([imgt.item(pix[0]-1, pix[1]-1), imgt.item(pix[0]-1, pix[1]), imgt.item(pix[0]-1, pix[1]+1), imgt.item(pix[0], pix[1]+1), imgt.item(pix[0]+1, pix[1]+1), imgt.item(pix[0]+1, pix[1]), imgt.item(pix[0]+1, pix[1]-1), imgt.item(pix[0], pix[1]-1)])

            if (2 <= np.sum(sub) <= 6 and
                np.sum(np.diff(sub)) <= 2 and
                not (sub.item(1) and sub.item(3) and sub.item(7)) and
                not (sub.item(1) and sub.item(5) and sub.item(7))):
                mark_for_deletion.append(npix)

        [imgt.itemset((list[n,0], list[n,1]), False) for n in mark_for_deletion]
        list = np.delete(list, mark_for_deletion, axis=0)

    return imgt

"""
Testing procedure
1. Load training data
2. Process input test data: obtain features of each object we want to identify
3. Predict each object using a specified method
"""
def test(img, setname="sans"):
    # load training features
    trainfeats = np.fromfile('train/' + setname + '.free')
    with open('train/' + setname + '.meta', 'r') as f:
        featdimensions = int(f.read())
    with open('train/order', 'r') as f:
        chars = f.read().replace('\n', '')

    trainfeats = trainfeats.reshape((trainfeats.size/(8*featdimensions**2), featdimensions, featdimensions, 8))

    # process test image
    testthin = thin(img)
    objlist = segment(testthin)
    cleanobjlist = preprocess(objlist)

    # attempt to output a letter for each object found in test image
    output = ""
    for obj in cleanobjlist:
        # extract features
        objfeat = freeman(obj, testthin, featdimensions, featdimensions)
        # calculate squared difference against each training feature
        sqdiff = np.sum(np.sum(np.sum(np.square(trainfeats - objfeat), axis=-1), axis=-1), axis=-1)
        output += chars[np.argmin(sqdiff)]

    return output

"""
Training procedure
Essentially this is the process of labelling our training data
Input an image. The image MUST CONTAIN all lower case letters, all upper case letters, and all digits in that order
The procedure then detects objects and assigns them letters/digits ("labels") based on this order
"""
def train(img, setname="sans", featdimensions=10):
    thinned = thin(img)
    objlist = segment(thinned)
    cleanobjlist = preprocess(objlist)

    # assign letters to features
    with open('train/order', 'r') as f:
        order = f.read().replace('\n', '')

    # get full features: I WANT EM ALL
    feats = []
    [feats.append(freeman(obj, thinned, featdimensions, featdimensions)) for obj in cleanobjlist]
    features = np.vstack([feats])

    #gnb_train(features)

    # write to external file so we don't need to retrain each time we attempt to recognize a font
    features.tofile('train/' + setname + '.free')
    with open('train/' + setname + '.meta', 'w') as f:
        f.write(str(featdimensions))

"""
Pre-processing
We try to clean the objects as much as possible:
1. Attempt to order the objects as a human would read them
2. Merge certain objects: the dots in 'i' and 'j', the "holes" in 'a', 'o', etc
"""
def preprocess(objlist):

    prefeats = prefeatextract(objlist)

    reordered_prefeats, reordered_objlist = reorder(prefeats)

    merged_objlist = mergeobj(reordered_prefeats, reordered_objlist)

    return merged_objlist

def prefeatextract(objlist):
    # get pre-features: we need absolute centers and heights, absolute top, and size of objects
    feats = []
    for obj in objlist:
        obj = np.array(obj)
        feats.append([obj[:,0].mean(), obj[:,1].mean(), obj[:,0].max() - obj[:,0].min(), obj[:,0].min(), obj.size])

    prefeats = np.array(feats)

    return prefeats

def reorder(prefeats):
    # figure out where lines change:
    # calc vertical differences between object centers
    # if vertical difference is above a certain threshold (in this case the mean height of objects)
    # we can be reasonably sure that we've changed lines
    # return the indices of new lines
    newlines = np.argwhere(np.diff(prefeats[:,0]) > np.mean(prefeats[:,2])).reshape(-1) + 1

    # obtain indices that would sort objects left to right for each newline
    lines = np.split(prefeats, newlines)
    objid = np.concatenate([np.argsort(lines[nline].T, axis=1)[1] + np.append(newlines, 0)[nline-1] for nline in range(len(lines))]).tolist()
    # sort objects
    orderedobjlist = [objlist[ind] for ind in objid]
    # sort prefeats
    prefeats[:] = prefeats[objid]

    return prefeats, orderedobjlist

def mergeobj(prefeats, orderedobjlist):
    # we try to merge objects if they're sufficiently close (for example the dots in 'i' and 'j')
    # here we merge objects that are 10x closer to each other than is expected
    # 10x is completely arbitrary
    delta = np.abs(np.diff(prefeats[:,1]))
    mergeat = np.argwhere(delta < delta.mean()*0.1).reshape(-1).tolist()

    # add "satellite" object to its "planet" object
    for ind in mergeat:
        orderedobjlist[ind] += orderedobjlist[ind+1]

    # remove "satellite" objects from object list
    mergedobjlist = [orderedobjlist[ind] for ind in range(len(orderedobjlist)) if (ind-1) not in mergeat]

    return mergedobjlist


"""
Thinning
Discard all pixels except for border pixels
"""
def thin(img, bg='dark'):
    imgb = np.copy(img)
    if img.dtype != bool:
        # binarize image
        imgb = binarize(img, bg)
    # copy binary image to new one that will contain our final image
    imgt = np.copy(imgb)

    # obtain image containing only border pixels:
    # for each (nonzero) pixel in the original image, check its neighbours
    # if NOT ALL of its neigbours are (nonzero) pixels, then it's a border pixel
    # if ALL of its neighbouts are (nonzero) pixels, then discard it
    # as we are setting pixels to zero, we need to output it to a new array: otherwise, our loop will detect false positives whenever it moves on to the next pixel
    sub = np.zeros((3,3), dtype=bool)

    for row in range(1, imgb.shape[0] - 1):
        for col in range(1, imgb.shape[1] - 1):
            if imgb.item(row,col):
                sub[:] = imgb[row-1:row+2, col-1:col+2]
                imgt.itemset((row,col), np.logical_not(np.all(sub)))

    """
    [imgt.itemset((row, col), np.logical_not(np.all(imgb[row-1:row+2, col-1:col+2]))) for row in range(1, imgb.shape[0] - 1) for col in range(1, imgb.shape[1] - 1) if imgb.item(row, col)]
    """

    return imgt

"""
Object detection
Attempt to cluster neighbouring pixels into separate objects
SERIOUSLY NEEDS OPTIMIZATION!
"""
def segment(img, minsize=0.1, chaincode=False):
    # use copy of img as we'll be eliminating elements
    imgt = np.copy(img)
    imgtb = imgt[1:-1, 1:-1]
    # obtain positions of border nonzeropix (non-zero elements)
    #nonzeropix = np.argwhere(imgtb) + np.ones((1,1), dtype=np.uint8)
    allobjpix = []
    allobjcc = []

    imgh = imgt.shape[0]*minsize

    startrow = 0
    # for all nonzero pixels: for all objects
    while np.any(imgt):
        # when we encounter a nonzero pixel we try to trace all nonzero pixels connected to it using depth
        # first search. in addition, this procedure erases pixels from the image once they're checked.
        # the search returns a list of connected pixel indices: an object

        startpix = findfirst(startrow, imgt)
        startrow = startpix[0]
        obj, cc = dfsi(imgt, startpix)
        # add object path to our output array
        if len(obj) > imgh:
            allobjpix += [obj]
            allobjcc += [cc]

        # each time an object is identified we reevaluate the indices of nonzero pixels in our image.
        # this is an expensive operation and should ideally be trashed: alternatively, we should delete
        # elements from nonzeropix directly once they've been accounted for in our dfs procedure
        #nonzeropix = np.argwhere(imgtb) + np.ones((1,1), dtype=np.uint8)

    if chaincode:
        return allobjpix, allobjcc
    else:
        return allobjpix

"""
Iterative depth-first search
"""
def dfsi(bitmap, startpixel):

    stack = [startpixel]
    objpix = [startpixel]
    objcc = []

    bound = False
    while stack:
        row, col = stack.pop()
        if row == 0 or col == 0: bound = True
        bitmap.itemset((row, col), False)
        edges = np.argwhere(bitmap[row-1:row+2, col-1:col+2]) - [1, 1]

        for edge in edges:
            nextpixel = [row+edge[0], col+edge[1]]
            if nextpixel not in objpix:
                stack += [nextpixel]
                objpix += [nextpixel]
                objcc += [[edge[0], edge[1]]]

    #[bitmap.itemset((pixel[0], pixel[1]), False) for pixel in objpix]

    if bound: return [], []
    else: return objpix, objcc

def findfirst(startrow, bitmap):
    while True:
        scanline = bitmap[startrow, :]
        if np.any(scanline):
            startcol = np.argwhere(scanline)[0,0]
            break
        startrow += 1

    return [startrow, startcol]

"""
Object image
Generates image matrix of specified object(s) from pixel positions of object(s)
FOR TESTING ONLY
"""
def getobjimg(objpix):
    objpix = np.asarray(objpix)
    if objpix.min() != 0:
        objpix[:] -= [objpix[...,0].min(), objpix[...,1].min()]

    objimg = np.zeros((objpix[...,0].max()+1, objpix[...,1].max()+1), dtype = int)

    for pix in objpix:
        objimg[pix[0], pix[1]] = True

    import matplotlib.pyplot as plt
    plt.imshow(objimg)
    plt.show()

def drawobj(img, objpix, box=True, opac=0.25):
    objpix = np.array(objpix)
    y = objpix[:,0]
    x = objpix[:,1]

    objimg = np.array(img*opac, dtype=np.uint8)
    objimg[y, x] = img[y, x]
    y0 = y.min()
    y1 = y.max()
    x0 = x.min()
    x1 = x.max()

    if box:
        green = [0,  255, 0]
        objimg[y0, x0:x1] = green
        objimg[y1, x0:x1] = green
        objimg[y0:y1, x0] = green
        objimg[y0:y1, x1] = green

    import matplotlib.pyplot as plt
    plt.imshow(objimg)
    plt.show()

"""
Extract Features according to UCI set
"""
def uci(objpix):
    # convert to numpy array
    objpix = np.array(objpix)

    y = objpix[...,0]
    x = objpix[...,1]

    ycent = y.mean()
    xcent = x.mean()

    # normalize
    if y.min() != 0 and x.min() != 0:
        objpix[:] -= [y.min(), x.min()]

    y = objpix[...,0]
    x = objpix[...,1]

    height = y.max()
    width = x.max()

    if height == 0 or width == 0: return [None, None, None, None]

    # calc ymean and xmean
    ymean = y.mean()/height
    xmean = x.mean()/width

    # calc yvar and xvar
    yvar = y.var()/height
    xvar = x.var()/width

    return np.array([ymean, xmean, yvar, xvar])

"""
Freeman chain encoding
source: http://www.codeproject.com/Articles/160868/A-C-Project-in-Optical-Character-Recognition-OCR-U
"""
def freeman(objpixels, img, ntracks=5, nsectors=5):
    # copy image so we don't end up overwriting it
    # (this isn't actually necessary)
    objimg = np.copy(img)

    # convert to numpy array
    objpix = np.array(objpixels)

    # get coordinates of center of mass of object
    ycent = objpix[...,0].mean()
    xcent = objpix[...,1].mean()

    # generate two arrays containing the distance and angle of each pixel relative to center of mass
    # get array of positions of all pixels relative to center of mass
    # this is a separate step cos we'll be reusing the position array
    position = objpix - [ycent, xcent]

    # get array of distances: distance is calculated through pythagoras' formula for triangles
    distance = np.sqrt(np.sum(np.square(position), axis=-1))

    # get array of angles: angle is obtained by calculating the inverse tanget of the y position against the x position
    # we need to consider the possibility of negative positions, hence we use the arctan2 numpy function
    # the output of arctan2 is [-pi, pi], so we divide this by pi to get the range [-1, 1]
    angles = np.arctan2(position[...,0], position[...,1])/np.pi

    # now we virtually "divide" our object into separate tracks and sectors
    # pixels are separated into different tracks based on their distance from the center
    # pixels are separated into different sectors based on the angle relative to the center
    distmax = distance.max()
    ftrack = distmax/ntracks
    rtracks = np.linspace(ftrack, distmax, ntracks)
    fsector = 2.0/nsectors - 1
    rsectors = np.linspace(fsector, 1, nsectors)

    chaincode = np.zeros((ntracks, nsectors, 8), dtype=int)
    npix = objpix.shape[0]
    for n in range(npix):
        track = np.argwhere(distance[n] <= rtracks)[0]
        sector = np.argwhere(angles[n] <= rsectors)[0]
        row = objpix.item(n, 0)
        col = objpix.item(n, 1)
        # find neighbouring pixels, and figure out which of them are nonzero
        relation = np.argwhere(np.delete(objimg[row-1:row+2, col-1:col+2].reshape(-1), 4))[:,0]
        for rel in relation:
            chaincode[track, sector, rel] += 1

    pernpix = 1.0/npix

    return chaincode*pernpix

"""
Gaussian Naive Bayes: continuous version of Naive Bayes classifier
Recall that Naive Bayes states that P(A|B) = P(A)*P(B|A)/P(B)
For the continuous version, each random variable/feature/attribute is assumed to be independent and distributed normally.
We are trying to achieve the (accurate) classification/prediction of the class of an object. In OCR, an "object" will be a collection of connected pixels and "class" a character we want to be able to predict ('a', '3', '@', etc); in other words we want to input a bunch of pixels and output a character.

A "feature" is equvalent to an "attribute" or "random variable" in this case. For freeman chain encoding,
a feature is the value contained in a single field of our track x sector x direction matrix.
A complete collection of the features of
"Features" refers to the collection of features for a single instance of an output class, for instance the
features of the output class "z" for a font "dejavusans".

A collection of all the features for all output classes is a "feature set",

We define a "training set" to be a collection of feature sets with each output class represented equally.
"""
def gnb_train(dataset='plat', feature_extraction='skeleton'):
    import os
    import matplotlib.image as mpimg

    # obtain the labels of our training dataset
    with open('train/order', 'r') as f:
        char = f.read().replace('\n', '')

    # obtain the actual training dataset
    files = [file for file in os.listdir('train/gnb/' + dataset) if file.startswith('font')]
    data = []

    # pandas: this specifies a multiindex from the combination of training labels and features
    ntracks, nsectors = 3, 3
    iter = pd.MultiIndex.from_product([[c for c in char], range(ntracks*nsectors*8)])

    # for each training instance we obtain the individual objects and extract features
    for file in files:
        img = mpimg.imread('train/gnb/'+ dataset + '/' + file)

        if feature_extraction == 'skeleton':
            bin = binarize(img, bg='light')
            pix, thinned = zhangsuen(np.argwhere(bin), bin)

        elif feature_extraction == 'skin':
            thinned = thin(img, bg='light')

        obj = preprocess(segment(thinned))
        feats = []
        [feats.append(freeman(o, thinned, ntracks, nsectors).reshape(-1)) for o in obj]
        pdfeats = pd.DataFrame(np.vstack([feats]).reshape((1, len(obj)*ntracks*nsectors*8)), index=[file], columns=iter)
        data.append(pdfeats)

    # collect features for all training instances and pickle
    df = pd.concat(data)
    df.to_pickle('train/gnb/model/plat_data')

    # collect means and variances between training instances for each feature for each output class and pickle
    pars = pd.concat((df.mean(), df.var()), axis=1).T
    pars.to_pickle('train/gnb/model/plat_pars')

def gnb_predict(img, bg='light', dataset='plat', feature_extraction='skeleton'):
    pars = pd.read_pickle('train/gnb/model/' + dataset + '_pars')
    ntracks, nsectors = 3, 3

    if feature_extraction == 'skeleton':
        bin = binarize(img, bg)
        npix = np.argwhere(bin)
        pix, thinned = zhangsuen(npix, bin)

    elif feature_extraction == 'skin':
        thinned = thin(img, bg)

    obj = preprocess(segment(thinned))
    feats = []
    [feats.append(freeman(o, thinned, ntracks, nsectors).reshape(-1)) for o in obj]
    feats = np.vstack([feats])

    string = ""
    for f in feats:
        prob = []
        [prob.append(np.exp(-0.5*np.square(f - pars.loc[0,c])/pars.loc[1,c])/np.sqrt(2*np.pi*pars.loc[1,c])) for c in pars.columns.levels[0]]
        arghyp = np.argmax([np.prod(p) for p in prob])
        string += pars.columns.levels[0][arghyp]

    return string

def preprocess2(path, bg='dark', fast=True, classifier='syntactic'):
    img = mpimg.imread(path)

    if fast:
        img = util.downsample(img)

    bin = binarize(img, bg)
    pix, skel = zhangsuen(np.argwhere(bin), bin)
    obj = preprocess(segment(skel))
