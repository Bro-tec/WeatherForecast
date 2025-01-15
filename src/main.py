from flet import *
import os
import math

from CollectData import get_learning_data as gld
import weather_lstm_pytorch as wlp


# chosen = []
chosenID = []

dd_stations = Dropdown(
    width=400,
    options=[
        dropdown.Option(str(id[1]["ID"] + ": " + id[1]["Name"]))
        for id in gld.load_stations_csv().iterrows()
    ],
)

dd_models = Dropdown(
    width=400,
    options=[
        dropdown.Option(str(str(ld)[:-4]))
        for ld in os.listdir("Models")
        if str(ld)[-4:] == ".pth"
    ],
)


def main(page: Page):
    page.scroll = ScrollMode.ALWAYS

    # def page_rezized(e):
    #     main_col.height = str(page.window.height - 100)
    #     page.update()

    # page.on_resize = page_rezized
    def resize_tab(e):
        if int(tabs.selected_index) == 0:
            main_col.height = 1000
        elif int(tabs.selected_index) == 1:
            main_col.height = 700
        else:
            main_col.height = 2000
        page.update()

    page.on_scroll = resize_tab

    def forecast_clicked(e):
        device = wlp.check_cuda()
        model, optimizer, history, others = wlp.load_own_Model(
            str(dd_models.value), device
        )
        wlp.forecast_weather(
            dd_models.value,
            hours_slider.value,
            chosenID,
            c_show_all.value,
            name=str(forecast_name.value),
            checks=[
                c_precipitation.value,
                c_precipitation_probability.value,
                c_precipitation_probability_6h.value,
                c_pressure_msl.value,
                c_temperature.value,
                c_sunshine.value,
                c_wind_speed.value,
                c_cloud_cover.value,
                c_dew_point.value,
                c_wind_gust_speed.value,
                c_relative_humidity.value,
                c_visibility.value,
                c_solar.value,
                c_wind_direction.value,
                c_wind_gust_direction.value,
                c_icon.value,
                c_condition.value,
                c_months.value,
                c_hours.value,
                c_pos.value,
            ],
            features=list(others["features"]),
        )
        # test_text.value = c_show_all.value
        forecast_images.controls.clear()
        for i in range(int(hours_slider.value)):
            forecast_images.controls.append(
                Image(
                    src=f"Forecasts/{forecast_name.value}_{i}.png",
                    width=724,
                    height=1024,
                    fit=ImageFit.NONE,
                    repeat=ImageRepeat.NO_REPEAT,
                    border_radius=border_radius.all(10),
                )
            )
        page.update()

    def create_model_clicked(e):
        if (
            len(str(model_name.value)) > 0
            and learning_rate_slider.value != 0
            and layer_slider.value != 0
            and hiddensize_slider.value != 0
            and sequence_slider.value != 0
            and dropout_slider.value != 0
        ):
            error_text.visible = False
            feature, indx, m, h, p = gld.create_feature(
                c_precipitation.value,
                c_precipitation_probability.value,
                c_precipitation_probability_6h.value,
                c_pressure_msl.value,
                c_temperature.value,
                c_sunshine.value,
                c_wind_direction.value,
                c_wind_speed.value,
                c_cloud_cover.value,
                c_dew_point.value,
                c_wind_gust_direction.value,
                c_wind_gust_speed.value,
                c_condition.value,
                c_relative_humidity.value,
                c_visibility.value,
                c_solar.value,
                c_icon.value,
                c_months.value,
                c_hours.value,
                c_pos.value,
            )
            ext = 0
            if m:
                ext += 1
            if h:
                ext += 1
            if p:
                ext += 3
            wlp.create_own_Model(
                str(model_name.value),
                int((len(feature) * (city_next_slider.value) + 1) + ext),
                len(feature),
                feature,
                indx,
                city_next=int(city_next_slider.value),
                learning_rate=learning_rate_slider.value,
                layer=int(layer_slider.value),
                hiddensize=int(hiddensize_slider.value),
                sequences=int(sequence_slider.value),
                dropout=dropout_slider.value,
                month=m,
                hours=h,
                position=p,
            )
            dd_models.options.append(dropdown.Option(str(model_name.value)))
        else:
            error_text.visible = True
            page.update()

    def train_model_switched(e):
        print("switch_train", switch_train.value)
        if switch_train.value:
            if (
                len(str(dd_models.value)) > 0
                and redo_slider.value != 0
                and epoch_slider.value != 0
                and batch_slider.value != 0
            ):
                # try:
                if len(str(skip_train.value)) <= 0:
                    skip_train.value = 0
                error_text.visible = False
                device = wlp.check_cuda()
                model, optimizer, history, others = wlp.load_own_Model(
                    str(dd_models.value), device
                )
                for train, label, i, r in gld.gen_trainDataHourly_Async(
                    skip_days=int(skip_train.value),
                    seq=others["sequences"],
                    max_batch=math.ceil(others["sequences"] / 24) * 24,
                    redos=int(redo_slider.value),
                    next_city_amount=others["city_next"],
                    feature_labels=list(others["features"]),
                    month=others["month"],
                    hours=others["hours"],
                    position=others["position"],
                ):
                    for epoch in range(int(epoch_slider.value)):
                        model, history, metrics, optimizer = wlp.train_LSTM(
                            str(dd_models.value),
                            train,
                            label,
                            model,
                            optimizer,
                            others["indx"],
                            history,
                            device,
                            epoch_count=int(epoch_slider.value),
                            batchsize=int(batch_slider.value),
                            cities_next=int(others["city_next"]),
                        )
                        wlp.save_own_Model(
                            str(dd_models.value), history, model, optimizer, device
                        )
                        wlp.plotting_hist(
                            history,
                            metrics,
                            str(dd_models.value),
                            min_amount=int(epoch_slider.value),
                            epoche=epoch,
                        )
                        train_images.controls.clear()
                        plt = ["accuracy", "loss", "matrix"]
                        for i in range(3):
                            forecast_images.controls.append(
                                Image(
                                    src=f"Plots/{str(dd_models.value)}_plot_{plt[i]}.png",
                                    width=724,
                                    height=1024,
                                    fit=ImageFit.NONE,
                                    repeat=ImageRepeat.NO_REPEAT,
                                    border_radius=border_radius.all(10),
                                )
                            )
                        page.update()
                        if not switch_train.value:
                            print(switch_train.value)
                            return
                # except ValueError:
                #     print(ValueError)
                #     error_text.visible = True
            else:
                error_text.visible = True

            page.update()

    def add_clicked(e):
        if dd_stations.value[:5] not in chosenID:
            chosenID.append(dd_stations.value[:5])
            # chosen.append(dd.value)
            lv.controls.append(Text(dd_stations.value, size=14))
            page.update()

    def sub_clicked(e):
        chosenID.pop()
        # chosen.pop()
        lv.controls.pop()
        page.update()

    def update_slider(e):
        learning_rate_text.value = (
            f"Set the learning rate : {learning_rate_slider.value:.5f}"
        )
        dropout_text.value = f"Set the dropout rate : {dropout_slider.value:.5f}"
        page.update()

    titel = Text("Weather Forecast", size=50)
    error_text = Text(
        "Error: Somewhere is an invalid value", size=50, color="red", visible=False
    )
    switch_train = Switch(label="Train", value=False, on_change=train_model_switched)
    b_add = ElevatedButton(text="Add", on_click=add_clicked)
    b_sub = ElevatedButton(text="Sub Last", on_click=sub_clicked)
    forecast_button = ElevatedButton(text="Forecast", on_click=forecast_clicked)
    create_model_button = ElevatedButton(
        text="Create Model", on_click=create_model_clicked
    )
    lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=True)
    hours_slider = Slider(min=0, max=1000, divisions=1000, label="{value} hours")
    epoch_slider = Slider(min=0, max=100, divisions=100, label="{value} epochs")
    batch_slider = Slider(min=0, max=10000, divisions=100, label="batchsize of {value}")
    city_next_slider = Slider(min=0, max=100, divisions=100, label="{value} cities")
    redo_slider = Slider(
        min=0,
        max=int(gld.load_stations_csv().shape[0] / 50),
        divisions=int(gld.load_stations_csv().shape[0] / 50),
        label="{value} times",
    )
    learning_rate_slider = Slider(
        min=0,
        max=0.2,
        divisions=2000,
        label="{value} learning rate",
        on_change=update_slider,
    )
    learning_rate_text = Text(
        f"Set the learning rate : {learning_rate_slider.value:.5f}", size=20
    )
    layer_slider = Slider(min=0, max=15, divisions=15, label="{value} layers")
    hiddensize_slider = Slider(
        min=0,
        max=1000,
        divisions=100,
        label="{value} hiddensize",
        on_change=update_slider,
    )
    sequence_slider = Slider(min=0, max=72, divisions=72, label="{value} sequences")
    dropout_slider = Slider(
        min=0, max=0.5, divisions=500, label="{value} dropout", on_change=update_slider
    )
    dropout_text = Text(f"Set the dropout rate : {dropout_slider.value:.5f}", size=20)
    c_show_all = Checkbox(
        label="show extra stations needed for forecasting", value=False
    )
    c_months = Checkbox(label="Months", value=True)
    c_hours = Checkbox(label="Hours", value=True)
    c_pos = Checkbox(label="Position(long+lat)", value=True)
    c_precipitation = Checkbox(label="precipitation", value=True)
    c_pressure_msl = Checkbox(label="pressure msl", value=True)
    c_sunshine = Checkbox(label="sunshine", value=True)
    c_temperature = Checkbox(label="temperature", value=True)
    c_wind_direction = Checkbox(label="wind direction", value=True)
    c_wind_speed = Checkbox(label="wind speed", value=True)
    c_cloud_cover = Checkbox(label="cloud cover", value=True)
    c_dew_point = Checkbox(label="dew point", value=True)
    c_relative_humidity = Checkbox(label="relative humidity", value=True)
    c_visibility = Checkbox(label="visibility", value=True)
    c_wind_gust_direction = Checkbox(label="wind_gust direction", value=True)
    c_wind_gust_speed = Checkbox(label="wind gust speed", value=True)
    c_condition = Checkbox(label="condition", value=True)
    c_precipitation_probability = Checkbox(
        label="precipitation probability", value=True
    )
    c_precipitation_probability_6h = Checkbox(
        label="precipitation probability 6h", value=True
    )
    c_solar = Checkbox(label="solar", value=True)
    c_icon = Checkbox(label="icon", value=True)
    forecast_name = TextField(label="Name for forecasts")
    model_name = TextField(label="Name your model")
    skip_train = TextField(label="Type the amount of days you want to skip")
    forecast_images = Row(expand=False, wrap=False, scroll=ScrollMode.ALWAYS)
    train_images = Row(expand=False, wrap=False, scroll=ScrollMode.ALWAYS)
    # chosens = Column([Text(ch, size=14) for ch in chosen])

    checks = Column(
        [
            Row(
                [
                    c_hours,
                    c_months,
                    c_pos,
                    c_precipitation,
                    c_precipitation_probability,
                    c_precipitation_probability_6h,
                    c_pressure_msl,
                    c_temperature,
                ]
            ),
            Row(
                [
                    c_sunshine,
                    c_wind_speed,
                    c_cloud_cover,
                    c_dew_point,
                    c_wind_gust_speed,
                    c_relative_humidity,
                    c_visibility,
                    c_solar,
                ]
            ),
            Row(
                [
                    c_wind_direction,
                    c_wind_gust_direction,
                    c_icon,
                    c_condition,
                ]
            ),
        ],
    )
    FW = Column(
        [
            Text("Choose the stations to forecast", size=20),
            dd_stations,
            Row([b_add, b_sub]),
            lv,
            Text("Choose the model", size=20),
            dd_models,
            Text("Choose the amount of hours", size=20),
            hours_slider,
            c_show_all,
            checks,
            Row([forecast_name, forecast_button]),
            error_text,
            forecast_images,
        ],
    )

    TM = Column(
        [
            Text("Choose your model", size=20),
            dd_models,
            skip_train,
            Text(
                "Choose how often it should load data to go through all cities (smaller number take longer but are prefered)",
                size=20,
            ),
            redo_slider,
            Text("Choose how many epoches the model should train", size=20),
            epoch_slider,
            Text("Choose how big the batchsize should be", size=20),
            batch_slider,
            error_text,
            switch_train,
            train_images,
        ]
    )
    CM = Column(
        [
            Text("Create your model", size=20),
            model_name,
            Text("Set the inputs", size=20),
            city_next_slider,
            checks,
            Text(f"Set the amount of layers", size=20),
            layer_slider,
            Text("Set the amount of hidden values", size=20),
            hiddensize_slider,
            Text("Set the amount of sequences(hours)", size=20),
            sequence_slider,
            learning_rate_text,
            learning_rate_slider,
            dropout_text,
            dropout_slider,
            create_model_button,
            error_text,
            # Text("Set the outputs", size=20),
        ],
    )

    tabs = Tabs(
        selected_index=0,
        animation_duration=400,
        tabs=[
            Tab(tab_content=Text("Create Model", size=20), content=Container(CM)),
            Tab(tab_content=Text("Train Model", size=20), content=Container(TM)),
            Tab(
                tab_content=Text("Forecast Weather", size=20),
                content=Container(content=FW),
            ),
        ],
    )

    main_col = Column([titel, tabs], height=1000, expand=True)

    page.add(
        main_col,
    )


app(main)
