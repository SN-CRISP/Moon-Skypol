import PySimpleGUI as sg
import numpy as np
import pandas as pd
import generic
import os.path


def tab_from_df(data):
    header = list(data.columns.values)

    rows = len(data.index)

    data_rows = []

    for i in range(0, rows):
        data_rows.append(data.iloc[i].to_numpy())

    return header, data_rows


def tab(fit_resume, result_par, erro, quality, band, condition, model, correction):
    header = ['Band', 'Parameters', 'Parameter errors', 'chi-square', 'reduced chi-aquare', 'BIC', 'Model',
              'Correction', 'Condition']
    df = pd.DataFrame(
        columns=['Band', 'Parameters', 'Parameter errors', 'chi-square', 'reduced chi-aquare', 'BIC', 'Model',
                 'Correction', 'Condition', 'Summary'])
    row = []
    bands = generic.listToString(band)
    i = 0
    if len(np.asarray(quality).shape) != 1:
        for j in range(0, len(band)):
            v = [band[j], np.array(result_par[j]), np.array(erro[j]), quality[j][0], quality[j][1], quality[j][2],
                 model, correction, condition]
            dta = fit_resume[fit_resume['BANDA'] == band[j]]
            row_df = {'Band': band[j], 'Parameters': np.array(result_par[j]), 'Parameter errors': np.array(erro[j]),
                      'chi-square': quality[j][0], 'reduced chi-aquare': quality[j][1],
                      'BIC': quality[j][2],
                      'Model': model, 'Correction': correction, 'Condition': condition, 'Summary': [dta]}
            df.loc[len(df) + i] = row_df
            i += 1
            row.append(v)
    else:
        v = [bands, np.array(result_par), np.array(erro), quality[0], quality[1], quality[2], model,
             correction, condition]
        row_df = {'Band': bands, 'Parameters': np.array(result_par), 'Parameter errors': np.array(erro),
                  'chi-square': quality[0], 'reduced chi-aquare': quality[1],
                  'BIC': quality[2],
                  'Model': model, 'Correction': correction, 'Condition': condition, 'Summary': [fit_resume]}
        df.loc[len(df)] = row_df
        row.append(v)

    tbl2 = sg.Table(values=row, headings=header,
                    auto_size_columns=True,
                    display_row_numbers=False,
                    justification='center', key='-TABLE-',
                    selected_row_colors='red on yellow',
                    enable_events=True,
                    expand_x=True,
                    expand_y=True,
                    enable_click_events=True, vertical_scroll_only=False)

    toprow, rows = tab_from_df(fit_resume)

    tbl1 = sg.Table(values=rows, headings=toprow,
                    auto_size_columns=True,
                    display_row_numbers=False,
                    justification='center', key='-TABLE-',
                    selected_row_colors='red on yellow',
                    enable_events=True,
                    expand_x=True,
                    expand_y=True,
                    enable_click_events=True, vertical_scroll_only=False)

    return tbl1, tbl2, df


def window_summary(fit_resume, result_par, erro, quality, band, condition, model, correction):
    tab1, tab2, df = tab(fit_resume, result_par, erro, quality, band, condition, model, correction)

    print(df)

    tab1_layout = [[tab1]]

    tab2_layout = [[tab2], [sg.Button('Save')]]

    layout = [
        [sg.TabGroup([[sg.Tab('Model', tab2_layout), sg.Tab('Data', tab1_layout, tooltip='tip')]], tooltip='TIP2')]]

    # Create the window
    window = sg.Window("SkyPol - Fit Summary", layout, size=(715, 300), auto_size_text=True, auto_size_buttons=True,
                       grab_anywhere=False, finalize=True, resizable=True)

    # Create an event loop
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if '+CLICKED+' in event:

            sg.popup("You clicked row:{} Column:{}".format(event[2][0], event[2][1]))

        if event == 'Save':
            models = pd.read_pickle('my_models.pkl')
            result = pd.concat([models, df], ignore_index=True, sort=False)
            result.to_pickle("my_models.pkl")

    window.close()
