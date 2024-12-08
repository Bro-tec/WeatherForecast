from PIL import Image, ImageDraw
import pandas as pd

im = Image.open("Landkarte_Deutschland.jpg")


def load_stations_csv():
    return pd.read_csv(
        "stations.csv", dtype={"ID": object}
    )  # otherwise lstm cant load this


def cropping(image, x_list, y_list):
    max_x = max(x_list) + 50
    min_x = min(x_list) - 50
    max_y = max(y_list) + 50
    min_y = min(y_list) - 50
    return image.crop((min_x, min_y, max_x, max_y))


def points(image, x_list, y_list):
    for x, y in zip(x_list, y_list):
        drw.ellipse(
            xy=(x - 3, y - 3, x + 3, y + 3),
            fill=(220, 20, 60),
        )
    return image


def l_to_px(list, rs, ps):
    min_l = min(list)
    max_l = max(list) - min_l
    print(min_l, max_l)
    print(list[0] - min_l)
    print((list[0] - min_l) / max_l)
    print(rs)
    list = [((x - min_l) / max_l * rs) + ps for x in list]
    return list


print(im.size)
stations = load_stations_csv()
y_l = stations["lat"].to_list()
y_l = [-x for x in y_l]
x_list = l_to_px(stations["lon"].to_list(), 648, 50)
y_list = l_to_px(y_l, 904, 60)
drw = ImageDraw.Draw(im, "RGB")
im = points(im, x_list, y_list)
# im = cropping(im, [300, 100], [500, 200])
im.show()
