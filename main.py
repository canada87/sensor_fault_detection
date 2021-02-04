#parametri globali:
#temperatura stanza
#velocita del rullo
#curvatura e lunghezza del mold
#composizione del steel
#presenza di stirrers

#parametri locali:
#temperatura in ogni settore
#pressione dell'acqua
#tipo di ugello

palette = ["#1F77B4","#FF7F0E","#2CA02C", "#00A3E0", '#4943cf', '#1eeca8', '#e52761', '#490b04', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b',
                   '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff',
                   '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4']

from AJ_draw import disegna as ds
from AJ_models_regression import learning_reg

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def work_function(train_X, train_y, test_X, test_Y, select_mod, verbose = 0, save = 'no', load = 'no', name = 'models'):
    try:
        col_operator = train_y.columns
        multi_target = True
    except:
        col_operator = train_y.name
        multi_target = False

    learn = learning_reg()
    if load == 'no':
        models = learn.get_models(select_mod)
        models, _ = learn.train_models(models, train_X, train_y)
        if save == 'yes':
            learn.save_model(models, name)
    else:
        models = learn.load_model(name)

    Predict_matrix = learn.predict_matrix_generator(models, test_X, multi_target = multi_target)

    if multi_target:
        for col in col_operator:
            st.header(col)
            y_test_target, y_pred_matrix_traget = learn.from_multilabel_to_single(test_Y, Predict_matrix, col)
            learn.score_models(y_test_target, y_pred_matrix_traget, verbose = 1)
            st.pyplot()

    else:
        learn.score_models(test_Y, Predict_matrix, verbose = verbose)
        st.pyplot()
    return learn.predict_matrix_generator(models, train_X, multi_target = multi_target), Predict_matrix

def deploy_function(train_X, train_y, select_mod, save = 'no', load = 'no', name = 'deploy_models'):
    try:
        col_operator = train_y.columns
        multi_target = True
    except:
        col_operator = train_y.name
        multi_target = False

    learn = learning_reg()
    if load == 'no':
        models = learn.get_models(select_mod)
        models, _ = learn.train_models(models, train_X, train_y)
        if save == 'yes':
            learn.save_model(models, name)
    else:
        models = learn.load_model(name)
    Predict_matrix = learn.predict_matrix_generator(models, train_X, multi_target = multi_target)

    return learn, models, Predict_matrix

df = pd.read_csv('steel_temperature.csv', delimiter=';', index_col=0)
####################################################################
####################################################################
df = df.sample(n=1000)
df = df.reset_index(drop=True)
####################################################################
####################################################################

ds().nuova_fig(1)
ds().titoli(titolo='4 random examples')
sample_plot = df.loc[:,['D'+str(i) for i in range(0,12)]].sample(n=4).reset_index(drop = True)
for i in range(4):
    ds().dati(x = np.arange(sample_plot.shape[1]), y=sample_plot.loc[i,:], colore= palette[i])
st.pyplot()

mod = st.sidebar.radio('modality', ['test','deploy'])
chain = st.sidebar.radio('Chain of models', ['no','yes'])
save_mod = st.sidebar.radio('save model', ['no','yes'])
load_mod = st.sidebar.radio('load model', ['no','yes'])
select_mod = st.sidebar.multiselect('which models', ['RandomForestRegressor', 'LinearRegression','ExtraTreesRegressor','DecisionTreeRegressor','ExtraTreeRegressor','BaggingRegressor'], default = ['RandomForestRegressor', 'LinearRegression'])

st.header('Elbow Method')
sse={}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df)
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
st.pyplot()

n_clusters = int(st.text_input('number of clusters', 4))
st.header('quality evaluation')
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(df)
df['Quantity'] = kmeans.predict(df)
st.write(df.head())
st.write(df.shape)

#rescale the temperature
temp_cols = []
for i in range(13):
    temp_cols.append('D'+str(i))
df[temp_cols] = df[temp_cols]/100

start = st.button('start')

#  ██████  ██████  ███    ██ ████████ ██████   ██████  ██      ██      ███████ ██████
# ██      ██    ██ ████   ██    ██    ██   ██ ██    ██ ██      ██      ██      ██   ██
# ██      ██    ██ ██ ██  ██    ██    ██████  ██    ██ ██      ██      █████   ██████
# ██      ██    ██ ██  ██ ██    ██    ██   ██ ██    ██ ██      ██      ██      ██   ██
#  ██████  ██████  ██   ████    ██    ██   ██  ██████  ███████ ███████ ███████ ██   ██

check_controll = st.checkbox('controller')
if check_controll and start:
    st.title('Training the Controller')
    st.subheader('The controller observes all the temperature along with the room temperature and the quality history and establishes the spead')
    st.latex(r'''T_1 T_2 T_3 T_4 T_{Room} Q = V''')
    col_controller = []
    for i in range(13):
        col_controller.append('D'+str(i))
    col_controller.append('vel')
    col_controller.append('T_room')
    col_controller.append('Quantity')
    df_controller = df.loc[:,col_controller]
    st.write(df_controller.head())
    st.write(df_controller.shape)

    df_controller_X = df_controller.drop(['vel'], axis = 1)
    df_controller_Y = df_controller['vel']

    controller_train_x, controller_test_x, controller_train_y, controller_test_y =  train_test_split(df_controller_X, df_controller_Y, test_size = 0.3, shuffle=False)
    controller_train_x, controller_test_x = controller_train_x.reset_index(drop = True), controller_test_x.reset_index(drop = True)
    controller_train_y, controller_test_y = controller_train_y.reset_index(drop = True), controller_test_y.reset_index(drop = True)

    if mod == 'test':
        controll_train, controll_test = work_function(controller_train_x, controller_train_y, controller_test_x, controller_test_y, select_mod, verbose = 1)
    else:
        controll_learn, controll_models, controll_pred = deploy_function(df_controller_X, df_controller_Y, select_mod, save = save_mod, load = load_mod, name = 'models/controller')

#  ██████  ██████  ███████ ██████   █████  ████████  ██████  ██████
# ██    ██ ██   ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
# ██    ██ ██████  █████   ██████  ███████    ██    ██    ██ ██████
# ██    ██ ██      ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
#  ██████  ██      ███████ ██   ██ ██   ██    ██     ██████  ██   ██

check_operator = st.checkbox('global operator')
if check_operator and start:
    st.title('Training the global operator')
    st.subheader('The global operator observes all the temperature along with the room temperature and the quality history and speed, and establishes the preassure on each section')
    st.latex(r'''T_1 T_2 T_3 T_4 V T_{Room} Q = P_1 P_2 P_3 P_4''')

    col_operator = []
    for i in range(1,13):
        col_operator.append('P'+str(i))

    df_operator_X = df.drop(col_operator, axis = 1)
    df_operator_Y = df.loc[:,col_operator]

    st.write(df_operator_X.head())
    st.write(df_operator_Y.head())

    operator_train_x, operator_test_x, operator_train_y, operator_test_y =  train_test_split(df_operator_X, df_operator_Y, test_size = 0.3, shuffle=False)
    operator_train_x, operator_test_x = operator_train_x.reset_index(drop = True), operator_test_x.reset_index(drop = True)
    operator_train_y, operator_test_y = operator_train_y.reset_index(drop = True), operator_test_y.reset_index(drop = True)

    if mod == 'test':
        if chain == 'yes':
            operator_train_x['vel'] = controll_train['Ensamble']
            operator_test_x['vel'] = controll_test['Ensamble']
        work_function(operator_train_x, operator_train_y, operator_test_x, operator_test_y, select_mod, verbose = 1)
    else:
        if chain == 'yes':
            df_operator_X['vel'] = controll_pred['Ensamble']
        operator_learn, operator_models, _ = deploy_function(df_operator_X, df_operator_Y, select_mod, save = save_mod, load = load_mod, name = 'models/operator')

# ███    ███  █████  ███████ ████████ ███████ ██████
# ████  ████ ██   ██ ██         ██    ██      ██   ██
# ██ ████ ██ ███████ ███████    ██    █████   ██████
# ██  ██  ██ ██   ██      ██    ██    ██      ██   ██
# ██      ██ ██   ██ ███████    ██    ███████ ██   ██

check_master = st.checkbox('master')
if check_master and start:
    st.title('master')
    st.subheader('the master observes all the temperature but the one related to its segment, the speed, quality and room temperature and establishes the temperature of its segment')
    st.write('the first temperature is known and it is not predicted')
    st.latex(r'''T_1 T_3 T_4 V T_{Room} Q = T_2''')

    col_operator = []
    for i in range(13):
        col_operator.append('D'+str(i))
    col_operator.append('vel')
    col_operator.append('T_room')
    col_operator.append('Quantity')

    df_master = df.loc[:,col_operator]
    st.write(df_master.head())
    st.write(df_master.shape)

    for i in range(1,13):
        if mod == 'test':
            st.header('D'+str(i))

        df_master_X = df_master.drop(['D'+str(i)], axis = 1)
        df_master_Y = df_master['D'+str(i)]

        master_train_x, master_test_x, master_train_y, master_test_y =  train_test_split(df_master_X, df_master_Y, test_size = 0.3, shuffle=False)
        master_train_x, master_test_x = master_train_x.reset_index(drop = True), master_test_x.reset_index(drop = True)
        master_train_y, master_test_y = master_train_y.reset_index(drop = True), master_test_y.reset_index(drop = True)

        if mod == 'test':
            if chain == 'yes':
                master_train_x['vel'] = controll_train['Ensamble']
                master_test_x['vel'] = controll_test['Ensamble']
            master_train, master_test = work_function(master_train_x, master_train_y, master_test_x, master_test_y, select_mod, verbose = 1)
        else:
            if chain == 'yes':
                df_master_X['vel'] = controll_pred['Ensamble']
            master_learn, master_models, master_pred = deploy_function(df_master_X, df_master_Y, select_mod, save = save_mod, load = load_mod, name = 'models/master_'+str(i))


# ███████ ██       █████  ██    ██ ███████
# ██      ██      ██   ██ ██    ██ ██
# ███████ ██      ███████ ██    ██ █████
#      ██ ██      ██   ██  ██  ██  ██
# ███████ ███████ ██   ██   ████   ███████
check_slave = st.checkbox('slave')
if check_slave and start:
    st.title('slave')
    st.subheader('the slave observe the incoming temperature, T room and speed and the desired output temperature and it establishes the preassure. Each segment works indipendently.')
    st.latex(r'''T_1 T_2 V T_{Room} = P_2''')


    for ii in range(0,12):
        col_operator = []
        col_operator.append('D'+str(ii))
        col_operator.append('D'+str(ii+1))
        col_operator.append('vel')
        col_operator.append('T_room')
        col_operator.append('P'+str(ii+1))

        df_slave = df.loc[:,col_operator]
        if mod == 'test':
            st.write(df_slave.head())
            st.write(df_slave.shape)
            st.header('P'+str(ii+1))

        df_slave_X = df_slave.drop(['P'+str(ii+1)], axis = 1)
        df_slave_Y = df_slave['P'+str(ii+1)]

        slave_train_x, slave_test_x, slave_train_y, slave_test_y =  train_test_split(df_slave_X, df_slave_Y, test_size = 0.3, shuffle=False)
        slave_train_x, slave_test_x = slave_train_x.reset_index(drop = True), slave_test_x.reset_index(drop = True)
        slave_train_y, slave_test_y = slave_train_y.reset_index(drop = True), slave_test_y.reset_index(drop = True)

        if mod == 'test':
            if chain == 'yes':
                slave_train_x['D'+str(ii+1)] = master_train['Ensamble']
                slave_test_x['D'+str(ii+1)] = master_test['Ensamble']
            work_function(slave_train_x, slave_train_y, slave_test_x, slave_test_y, select_mod, verbose = 1)
        else:
            if chain == 'yes':
                df_slave_X['vel'] = master_pred['Ensamble']
            slave_learn, slave_models, _ = deploy_function(df_slave_X, df_slave_Y, select_mod, save = save_mod, load = load_mod, name = 'models/slave_'+str(ii))


# ███████ ██    ██ ██████  ███████ ██████  ██    ██ ██ ███████  ██████  ██████
# ██      ██    ██ ██   ██ ██      ██   ██ ██    ██ ██ ██      ██    ██ ██   ██
# ███████ ██    ██ ██████  █████   ██████  ██    ██ ██ ███████ ██    ██ ██████
#      ██ ██    ██ ██      ██      ██   ██  ██  ██  ██      ██ ██    ██ ██   ██
# ███████  ██████  ██      ███████ ██   ██   ████   ██ ███████  ██████  ██   ██

check_supervisor = st.checkbox('supervisor')
if check_supervisor and start:
    st.title('supervisor')
    st.subheader('the supervisor takes all temperatures, the quality and the room temperature and establishes the speed and all the pressures')
    st.write('the supervisor works alone, does not need any submitted model')
    st.latex(r'''T_1 T_2 T_3 T_4 T_{Room} Q = P_1 P_2 P_3 P_4 V''')


    col_y = ['P'+str(i) for i in range(1,13)]
    col_y.append('vel')

    df_supervisor_X = df.drop(col_y, axis = 1)
    df_supervisro_Y = df.loc[:,col_y]

    st.write(df_supervisor_X.head())

    supervisor_train_x, supervisor_test_x, supervisor_train_y, supervisor_test_y =  train_test_split(df_supervisor_X, df_supervisro_Y, test_size = 0.3, shuffle=False)
    supervisor_train_x, supervisor_test_x = supervisor_train_x.reset_index(drop = True), supervisor_test_x.reset_index(drop = True)
    supervisor_train_y, supervisor_test_y = supervisor_train_y.reset_index(drop = True), supervisor_test_y.reset_index(drop = True)
    if mod == 'test':
        work_function(supervisor_train_x, supervisor_train_y, supervisor_test_x, supervisor_test_y, select_mod, verbose = 1)
    else:
        supervisor_learn, supervisor_models, _ = deploy_function(df_supervisor_X, df_supervisro_Y, select_mod, save = save_mod, load = load_mod, name = 'models/supervisor')



# ██████  ███████ ██████  ██       ██████  ██    ██
# ██   ██ ██      ██   ██ ██      ██    ██  ██  ██
# ██   ██ █████   ██████  ██      ██    ██   ████
# ██   ██ ██      ██      ██      ██    ██    ██
# ██████  ███████ ██      ███████  ██████     ██
st.title('Deploy')
st.write('Three network are userd to control the process:')
st.subheader('Supervisor')
st.subheader('Controller -> global operator')
st.subheader('Controller -> Master -> Slave')

T_prova = [1299, 1287, 1269, 1276, 1206, 1232, 1267, 1232, 1253, 1217, 1273,  1244, 1211]

if mod == 'deploy':
    T = [0 for i in range(13)]
    T[0] = float(st.text_input('T0',T_prova[0]))
    T[1] = float(st.text_input('T1',T_prova[1]))
    T[2] = float(st.text_input('T2',T_prova[2]))
    T[3] = float(st.text_input('T3',T_prova[3]))
    T[4] = float(st.text_input('T4',T_prova[4]))
    T[5] = float(st.text_input('T5',T_prova[5]))
    T[6] = float(st.text_input('T6',T_prova[6]))
    T[7] = float(st.text_input('T7',T_prova[7]))
    T[8] = float(st.text_input('T8',T_prova[8]))
    T[9] = float(st.text_input('T9',T_prova[9]))
    T[10] = float(st.text_input('T10',T_prova[10]))
    T[11] = float(st.text_input('T11',T_prova[11]))
    T[12] = float(st.text_input('T12',T_prova[12]))
    T_room = float(st.text_input('TRoom',25))
    Q = float(st.text_input('Quality',0))

    T = np.array(T)/100
    T = T.tolist()
    T.append(T_room)
    T.append(Q)

    cols = ['T'+str(i) for i in range(13)]
    cols.append('TRoom')
    cols.append('Quality')
    querry = pd.DataFrame(T).T
    querry.columns = cols

    if check_supervisor:
        Predict_matrix = supervisor_learn.predict_matrix_generator(supervisor_models, querry, multi_target = True)
        st.write(Predict_matrix['Ensamble'])

    elif check_controll and check_operator:
        Predict_matrix = controll_learn.predict_matrix_generator(controll_models, querry, multi_target = False)
        st.write('vel')
        st.write(Predict_matrix['Ensamble'])
        querry['vel'] = Predict_matrix['Ensamble']
        cols.pop(-1)
        cols.pop(-1)
        cols.append('vel')
        cols.append('TRoom')
        cols.append('Quality')
        querry = querry[cols]
        Predict_matrix = operator_learn.predict_matrix_generator(operator_models, querry, multi_target = True)
        st.write('preassures')
        st.write(Predict_matrix['Ensamble'])

    elif check_controll and check_master and check_slave:
        Predict_matrix = controll_learn.predict_matrix_generator(controll_models, querry, multi_target = False)
        st.write('vel')
        st.write(Predict_matrix['Ensamble'])
        querry['vel'] = Predict_matrix['Ensamble']
        cols.pop(-1)
        cols.pop(-1)
        cols.append('vel')
        cols.append('TRoom')
        cols.append('Quality')
        querry = querry[cols]
        slave_results = pd.DataFrame()

        for i in range(1,13):
            querry_master = querry.drop(['T'+str(i)], axis = 1)
            Predict_matrix = master_learn.predict_matrix_generator(master_models, querry_master, multi_target = False)
            querry_slave = querry_master.loc[0, ['T'+str(i-1), 'vel', 'TRoom']]
            querry_slave['T'+str(i)] = Predict_matrix['Ensamble'].values[0]
            cols_new = ['T'+str(i-1), 'T'+str(i), 'vel', 'TRoom']
            querry_slave = querry_slave[cols_new]
            querry_slave = pd.DataFrame(querry_slave).T
            Predict_matrix = slave_learn.predict_matrix_generator(slave_models, querry_slave, multi_target = False)
            slave_results['P'+str(i)] = Predict_matrix['Ensamble']

        st.write('preassures')
        st.write(slave_results)
