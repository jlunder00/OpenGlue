import openslide

def main():
    slide = openslide.OpenSlide('/host_Data/Data/MegaDepth/BreastImages/Breast-1/Breast-1_Ki67.svs')

    down1 = slide.level_downsamples[1]
    dims = slide.dimensions

    newdims = tuple((x/(down1-0.0001)) for x in dims)
    print(newdims)

    image = slide.get_thumbnail(newdims)
    image.save('/host_Data/Data/MegaDepth/BreastImages/thumb.png', 'png')

main()
