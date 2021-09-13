# resize 1
from uwimg import *
im = load_image("data/dogsmall.jpg")
a = nn_resize(im, im.w*4, im.h*4)
save_image(a, "dog4x-nn")

#resize 2
im = load_image("data/dogsmall.jpg")
a = bilinear_resize(im, im.w*4, im.h*4)
#save_image(a, "edited-dog4x-bl")
save_image(a, "dog4x-bl")

#resize 3
im = load_image("data/dog.jpg")
a = nn_resize(im, im.w//7, im.h//7)
save_image(a, "edited")
#save_image(a, "dog7th-bl")

#filtering
im = load_image("data/dog.jpg")
f = make_box_filter(7)
blur = convolve_image(im, f, 1)
thumb = nn_resize(blur, blur.w//7, blur.h//7)
save_image(thumb, "dogthumb")

#gaussian filter
im = load_image("data/dog.jpg")
f = make_gaussian_filter(2)
blur = convolve_image(im, f, 1)
save_image(blur, "dog-gauss2")

#hybrid
im = load_image("figs/ronbledore.jpg")
f = make_gaussian_filter(2)
lfreq = convolve_image(im, f, 1)
hfreq = im - lfreq
reconstruct = lfreq + hfreq
save_image(lfreq, "ron-low-frequency")
save_image(hfreq, "ron-high-frequency")
save_image(reconstruct, "ron-reconstruct")

#feature normalization
im = load_image("data/dog.jpg")
res = sobel_image(im)
mag = res[0]
feature_normalize(mag)
save_image(mag, "magnitude")

#colorized sobel
im = load_image("data/dog.jpg")
res = colorize_sobel(im)
save_image(res, "sobel")
