a = v2.Resize(
    size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True
)
b = a(img)
