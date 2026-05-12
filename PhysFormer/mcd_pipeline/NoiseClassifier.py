import pickle
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import biosppy.signals.ecg as ecg
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, f1_score, roc_curve, make_scorer
from scipy import io, signal
from ecgdetectors import Detectors
from mcd_pipeline import utils as u

import matplotlib.pyplot as plt


class NoiseClassifier:
    
    def __init__(self, model_path = None, scaler_path = None, attributes_path = None, fs = 300, attributes = ['bSQI', 'sSQI', 'kSQI', 'pSQI', 'basSQI', 'fSQI']) -> None:
        if model_path != None:
            with open(model_path, 'rb') as model_file:
                self.model =  pickle.load(model_file)
            with open(scaler_path, 'rb') as scaler_file:
                self.scaler = pickle.load(scaler_file)
            with open(attributes_path, 'rb') as attributes_file:
                self.attributes = pickle.load(attributes_file)
        else:
            self.model = None 
            self.scaler = None
            self.attributes = attributes

        self.fs = fs
        self.all_attributes = ['bSQI', 'sSQI', 'kSQI', 'pSQI', 'basSQI', 'fSQI']
                 
    def train_model(self, path, clf = 'KNN'):
        df_attributes_windows = pd.read_csv(path).drop(['Unnamed: 0'], axis=1)
        df_attributes_windows = df_attributes_windows.sample(frac = 1, random_state = 5)
        X_train = df_attributes_windows[self.attributes]
        y_train = df_attributes_windows['Ritmo']
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)

        if clf == 'KNN':
            model = KNeighborsClassifier(n_neighbors = 8, weights = 'distance', p = 1)
        elif clf == 'SVM':
            model = svm.SVC(kernel='rbf', C = 1, gamma = 1, probability = True)
        elif clf == 'LR':
            model = LogisticRegression(penalty = 'l1', C = 1, solver = 'liblinear', max_iter=100)
        elif clf == 'XGB':
            model = XGBClassifier(learning_rate = 0.01, max_depth = 7, n_estimators = 200)
 
        model.fit(X_train_s, y_train)
        self.model = model
        self.scaler = scaler
    
    def save_model(self, filename = 'trained_noise_model'):
        filename_model = filename +'.sav'
        filename_scaler = filename + '_scaler.sav'
        filename_attributes = filename + '_attributes.sav'
        with open(filename_model, 'wb') as f:
            pickle.dump(self.model, f)
        with open(filename_scaler, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(filename_attributes, 'wb') as f:
            pickle.dump(self.attributes, f)
        
        
    def classify(self, signal, length_wind = 5, med = True, ponderado = 3, threshold = 0.5, bp = True):
        n_signal = len(signal)
        n_ventana = self.fs*length_wind
        k = int((n_signal - n_ventana)/self.fs)
        signal_pb = u.pasabanda(u.correct_signal(u.med_filt(signal, self.fs), self.fs), self.fs)     # Lo cambie miercoles de noche!! Antes no tenia correct_signal y med_filt
        detectors = Detectors(self.fs)
        i_peaks_H = detectors.hamilton_detector(signal_pb)
        i_peaks_WQRS = detectors.wqrs_detector(signal_pb)
        
        if med == True:
            signal = u.med_filt(signal, self.fs)

        signal = u.correct_signal(signal, self.fs)
        signal_bp = u.pasabanda(signal, self.fs)
        signal_lp = self.__pasabajos__(signal, cut = 40)
        X = []
        for i in range(k):
            start = i*self.fs
            window_lp = signal_lp[start:(start+n_ventana)]
            if bp:
                window_bp = signal_bp[start:(start+n_ventana)]
            else:
                window_bp = window_lp      
            s, k, p, bas, f = self.__extractor_carac__(window_bp = window_bp, window_lp = window_lp)
            b = self.__calculador_bSQI_ventana__(i_peaks_H, i_peaks_WQRS, start)
            if np.isnan([s, k, p]).any():
                print('nan')
                x = [b] + x[1:]
            else:
                x = [b, s, k, p, bas, f]
            X.append(x)

        X_test = pd.DataFrame(X, columns = self.all_attributes)
        X_test = X_test[self.attributes]
        X_test = self.scaler.transform(X_test)
        y_prob = self.model.predict_proba(X_test)[:,1]

        y_aux = np.zeros(int(n_signal/self.fs), dtype = 'float')
        
        if length_wind == 5:
            y_aux[0] = y_prob[0]
            y_aux[1] = (2*y_prob[0] + y_prob[1]) / 3
            y_aux[2] = (y_prob[2] + 2*y_prob[1] + ponderado*y_prob[0]) / (3+ponderado)
            y_aux[3] = (y_prob[1]*ponderado + 2*y_prob[0] +2*y_prob[2] + y_prob[3] )/(5+ponderado)
            y_aux[-4] = (y_prob[-2]*ponderado + 2*y_prob[-1] +2*y_prob[-3] + y_prob[-4] )/(5+ponderado)
            y_aux[-3] = (y_prob[-3] + 2*y_prob[-2] + ponderado*y_prob[-1]) / (3+ponderado)
            y_aux[-2] = (y_prob[-2] + 2*y_prob[-1]) / 3
            y_aux[-1] = y_prob[-1]

            y_ruido = np.zeros(int(n_signal/self.fs), dtype = 'int')
            y_ruido[0] = u.round_prob(y_aux[0], threshold = threshold)
            y_ruido[1] = u.round_prob(y_aux[1], threshold = threshold)
            y_ruido[2] = u.round_prob(y_aux[2], threshold = threshold)
            y_ruido[3] = u.round_prob(y_aux[3], threshold = threshold)
            y_ruido[-4] = u.round_prob(y_aux[-4], threshold = threshold)
            y_ruido[-3] = u.round_prob(y_aux[-3], threshold = threshold)
            y_ruido[-2] = u.round_prob(y_aux[-2], threshold = threshold)
            y_ruido[-1] = u.round_prob(y_aux[-1], threshold = threshold)
            y_aux[2:-3] = y_prob


            for i in range(4, len(y_ruido)-4):
            
                y_prev_prev = y_aux[i-2]
                y_prev = y_aux[i-1]
                y_main = y_aux[i]
                y_next = y_aux[i+1]
                y_next_next = y_aux[i+2]

                prob = (y_prev_prev + 2*y_prev + ponderado*y_main + 2*y_next + y_next_next) / (6 + ponderado)

                y_ruido[i] = u.round_prob(prob, threshold = threshold)

        elif length_wind == 3:

            y_aux[0] = y_prob[0]
            y_aux[1] = (2*y_prob[0] + y_prob[1]) / 3
            y_aux[-2] = (y_prob[-2] + 2*y_prob[-1]) / 3
            y_aux[-1] = y_prob[-1]
            
            y_ruido = np.zeros(int(n_signal/self.fs), dtype = 'int')
            y_ruido[0] = u.round_prob(y_aux[0], threshold = threshold)
            y_ruido[1] = u.round_prob(y_aux[1], threshold = threshold)
            y_ruido[-2] = u.round_prob(y_aux[-2], threshold = threshold)
            y_ruido[-1] = u.round_prob(y_aux[-1], threshold = threshold)

            y_aux[1:-2] = y_prob

            for i in range(2, len(y_ruido)-2):
                
                y_prev = y_aux[i-1]
                y_main = y_aux[i]
                y_next = y_aux[i+1]
                prob = (1*y_prev + 2*y_main + 1*y_next) /(4)
                
                y_ruido[i] = u.round_prob(prob, threshold = threshold)

        return y_ruido
    
    def cut_noise(self, signal, med = True, length_wind = 5, threshold = 0.5):
        resultado = self.classify(signal, length_wind, med, threshold = threshold)
        if med == True:
            signal = u.med_filt(signal, self.fs)
        signal = u.correct_signal(signal, self.fs)
        senal_cortada = []
        segmentos = []
        inicios = []
        aux = 0
        inicio = 0
        final = 0
        cont = 0
        primero = True
        for j  in range(len(resultado)):
            segundo = resultado[j]
        
            if segundo != 0 and aux == 1:
                final = j
            
                aux = 2
                cont = 0

                if inicio == final:
                    senal_cortada = senal_cortada + list(signal[j*self.fs])
                else:
                    senal_cortada = senal_cortada + list(signal[inicio*self.fs:final*self.fs])
                    segmento = signal[inicio*self.fs:final*self.fs]
                    # if len(segmento) > 5*fs:
                    segmentos.append(segmento)
                    inicios.append(int(inicio*self.fs))

            if ((aux == 2) or (segundo != 0 and primero)):
                cont+=1
                if j != (len(resultado)-1):
                    siguiente = resultado[j+1]
                else:
                    siguiente = 0

                if siguiente == 0 or (primero and resultado[j+1]==0):
                    aux = 0
                    senal_cortada = senal_cortada + list(np.zeros(cont*self.fs))

            if segundo == 0 and aux == 0:
                inicio = j
                aux = 1
                if primero:
                    primero = False
            
            if j == (len(resultado)-1) and segundo == 0:
                senal_cortada = senal_cortada + list(signal[inicio*self.fs:])
                segmento = signal[inicio*self.fs:]
                segmentos.append(segmento)
                inicios.append(int(inicio*self.fs))

        return segmentos, inicios, senal_cortada, resultado
 
    def test_model(self, path, clf = 'LR', train = True, n_neighbors = 8, weights = 'distance', p_KNN = 1):
        df_attributes_windows = pd.read_csv(path).drop(['Unnamed: 0'], axis=1)
        df_attributes_windows = df_attributes_windows           # .sample(frac = 1, random_state= 5)
        df_X = df_attributes_windows[self.attributes]
        df_y = df_attributes_windows['Ritmo']

        if train == True:
            scaler = StandardScaler()
            scaler.fit(df_X)
            df_X_s = scaler.transform(df_X)
            X_train, X_test, y_train, y_test = train_test_split(df_X_s, df_y, test_size = 0.2, random_state = 212)
            if clf == 'KNN':
                model = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, p = p_KNN)
            elif clf == 'SVM':
                model = svm.SVC(kernel='rbf', C= 1)
            elif clf == 'LR':
                model = LogisticRegression(max_iter=100)
            elif clf == 'XGBoost':
                model = XGBClassifier()
                
            model.fit(X_train, y_train)

            cv_scores = {}
            specificity = make_scorer(recall_score, pos_label=0)
            sensitivity = make_scorer(recall_score, pos_label=1)
            cv_scores['Accuracy cv'] = round(np.average(cross_val_score(model, X_train, y_train, cv=5))*100, 2)
            cv_scores['Precision cv'] = round(np.average(cross_val_score(model, X_train, y_train, cv=5, scoring = 'precision'))*100, 2)
            cv_scores['Specificity cv'] = round(np.average(cross_val_score(model, X_train, y_train, cv=5, scoring = specificity))*100, 2)
            cv_scores['Sensitivity cv'] = round(np.average(cross_val_score(model, X_train, y_train, cv=5, scoring = sensitivity))*100, 2)
            cv_scores['F1-score cv'] = round(np.average(cross_val_score(model, X_train, y_train, cv=5, scoring = 'f1'))*100, 2)

            log_loss = np.average(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_log_loss'))

        else:
            model = self.model
            scaler = self.scaler
            X_test = scaler.transform(df_X)
            y_test = np.array(df_y)
            cv_scores = None

        y_pred = model.predict(X_test)
        df_attributes_windows = df_attributes_windows.sample(frac = 1, random_state = 6)
        df_X = df_attributes_windows[self.attributes]
        df_y = df_attributes_windows['Ritmo']

        conf_matrix = confusion_matrix(y_test, y_pred, labels = [0,1])
        tn, fp, fn, tp = conf_matrix.ravel()
        accu = (tn + tp)*100 / (tn + fp + fn + tp)
        prec = tp*100 / (tp + fp)
        spec = tn*100 / (tn+fp)
        sens = tp*100 / (tp+fn)
        scores = {'Accuracy': round(accu, 3), 'Precision': round(prec, 3), 'Specificity': round(spec, 3), 'Sensitivity': round(sens, 3)}
    
        return y_pred, y_test, scores, conf_matrix, cv_scores, log_loss

    def save_attributes(self, df_windows, filename, fs = None, path_b = 'bSQI_ventanas_pb.csv', med = True, bp = True):
        if fs == None:
            fs = self.fs

        if 'Ritmo' in df_windows.columns:
            ritmos = df_windows['Ritmo']
            windows = df_windows.drop('Ritmo', axis = 1)

        bs = (pd.read_csv(path_b).drop('Unnamed: 0', axis = 1))['bSQI']
        X_test = []
        
        for i in range(len(windows)):
            window = np.array(windows.iloc[i])[:fs*3]
            if med:
                window = u.med_filt(window, fs)
            window = u.correct_signal(window, fs)
            window_lp = self.__pasabajos__(window)
            if bp:
                window_bp = u.pasabanda(window, self.fs)
            else:
                window_bp = window_lp
            s, k, p, bas, f = self.__extractor_carac__(window_bp, window_lp) 
            b = bs[i]

            if 'Ritmo' in df_windows.columns:
                R = ritmos[i]
                column_names = ['Ritmo'] + self.all_attributes

                x = [R, b, s, k, p, bas, f]
            else:
                column_names = self.all_attributes
                x = [b, s, k, p, bas, f]
            if np.isnan([s, k, p]).any() == False:
                X_test.append(x)

        df_X = pd.DataFrame(X_test, columns = column_names)

        filename = 'Dataframe_atributos_ruido_' + filename + '.csv'
        df_X.to_csv(filename)

        return
    
    def plot_signal(self, senal_, fs =  None, peaks = True, r_peaks_ = None, r_peak_num = True, title = None, med_correct = False, clf_noise = None, 
            AF_prob = None, legends = True, r_peak_loc = 'zero', legend_loc = 'lower right', y_lim = None, x_lim = None,
            k_fig_height = 1, k_fig_len = 1, res = False, res_loc = 1, ect = None, fig_path = None):
        if not fs:
            fs = self.fs
        senal = senal_
        if med_correct:
            senal = u.correct_signal(u.med_filt(senal, fs), fs)
            label_senal = 'Señal preprocesada'
        else:
            label_senal = 'Señal'
        dur = len(senal) / fs
        t_senal = np.linspace(0, dur, len(senal))
        f_size = int(dur * 20/30)
        
        x = []

        fig = plt.figure(figsize = [f_size*k_fig_len/1.5,5*k_fig_height/1.5])

        if clf_noise == None:
            plt.plot(t_senal, senal, label = label_senal)
            r_peaks = u.R_peaks(senal, fs)
            x = t_senal[r_peaks]
            n = 1

        else:
            segmentos, inicios, senal_cortada, resultado = clf_noise.cut_noise(senal_)
            n = len(segmentos)
            print('n segmentos:', n)
            
            r_peaks = []

            if AF_prob != None:
                label_AF = 'P(FA) clf bal:      ' + str(round(AF_prob[0], 2))
                plt.plot(np.average(t_senal), np.average(senal), 'w', label = label_AF)
                label_AF = 'P(FA) clf no bal: ' + str(round(AF_prob[1], 2))
                plt.plot(np.average(t_senal), np.average(senal), 'w', label = label_AF)
            
            if n > 0:
                start_0 = inicios[0]
                if start_0 != 0:
                    t_0 = np.linspace(0, start_0/fs, start_0)
                    plt.plot(t_0, senal[:start_0], 'tab:red', label = 'Segmento ruidoso')
                for j in range(n):
                    segmento = segmentos[j]
                    start_seg = inicios[j]
                    end_seg = start_seg + len(segmento)
                    t_seg = np.linspace(start_seg/fs, end_seg/fs, len(segmento))
                    # plt.plot(t_seg, segmento, 'tab:blue')

                    if j == 0:  
                        plt.plot(t_seg, senal[start_seg:end_seg], 'tab:blue', label = 'Segmento no ruidoso')
                    else:
                        plt.plot(t_seg, senal[start_seg:end_seg], 'tab:blue')

                    r_peaks_seg = u.R_peaks(segmento, fs, PyT = r_PyT)
                    r_peaks = r_peaks + list(start_seg + r_peaks_seg)
                    x = x + list(t_seg[r_peaks_seg])

                    if end_seg < len(senal):
                        if (j+1) < n:
                            end_n = inicios[j+1]
                        else:
                            end_n = len(senal)
                        start_n = end_seg
                        t_n = np.linspace(start_n/fs, end_n/fs, end_n - start_n)
                        if (j == 0) and (start_0 == 0):
                            plt.plot(t_n, senal[start_n:end_n], 'tab:red', label = 'Segmento ruidoso')
                        else:
                            plt.plot(t_n, senal[start_n:end_n], 'tab:red')

        if (peaks == True) and (len(x) > 0):
            if r_peaks_ != None:
                r_peaks = r_peaks_
                x = t_senal[r_peaks]
            p = np.arange(1,len(x)+1)
            if r_peak_loc == 'zero':
                y = np.zeros(len(x))
            elif r_peak_loc == 'peak':
                y = senal[r_peaks]
            elif r_peak_loc == 'abs':
                y = np.abs(senal[r_peaks])
            plt.plot(x, y, 'oy', label = 'Picos R')
            if r_peak_num:
                for i, txt in enumerate(p):
                    plt.annotate(txt, (x[i], y[i]))

        if (clf_noise != None) and res:
            p = resultado
            x = np.linspace(0.35, dur - 0.65, len(p))
            y = np.ones(len(p))*np.average(senal[r_peaks])*2*res_loc
            plt.plot(x, y, 'ow')
            for i, txt in enumerate(p):
                plt.annotate(txt, (x[i], y[i]))

        if ect != None:
            t_ect = ect/fs
            plt.plot([t_ect], [senal[ect]], 'or', label = 'Ectópico')

        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        if x_lim != None:
            plt.xlim(x_lim)
            dur = x_lim[-1]
        else:
            plt.xlim([0, dur])
        t_ticks = np.arange(0,dur+1,1)
        plt.xticks(t_ticks)
        if y_lim != None:
            plt.ylim(y_lim)
        plt.grid(visible=None, which='major', axis='both')

        if title != None:
            plt.title(title)
        if legends and (n > 0):
            plt.legend(loc = legend_loc, fontsize='x-small')

        if fig_path:
            plt.subplots_adjust(bottom=0.2)
            image_format = 'svg' # e.g .png, .svg, etc.
            fig.savefig(fig_path, format=image_format, dpi=1200)

        plt.show()

        return
    
#---------------------PRIVATE CLASS FUNCTIONS TO CALCULATE SIGNALS QUALITY INDEXES-------------------------------------------------------------------------      
    
    def __N_matches__(self, indexes_WQRS, indexes_H):
        tol = int(self.fs/6)
        N = 0
        for i in range(len(indexes_H)):
            i_H = indexes_H[i]
            i_sim = indexes_WQRS[0]
            for j in range(len(indexes_WQRS)):
                i_WQRS = indexes_WQRS[j]
                if abs(i_H - i_WQRS) < abs(i_H - i_sim):
                    i_sim = i_WQRS
                
            if i_sim - tol <= i_H <= i_sim + tol:
                N = N + 1
        
        return N

    def __calculador_bSQI_ventana__(self, i_H, i_WQRS, inicio):
        fin = inicio + 5*self.fs

        i_H = [i_peak for i_peak in i_H if (inicio <= i_peak <= fin)] 
        i_WQRS = [i_peak for i_peak in i_WQRS if (inicio <= i_peak <= fin)] 

        N_H = len(i_H)
        N_WQRS = len(i_WQRS)
        if (N_H*N_WQRS != 0):
            N_matched = self.__N_matches__(i_WQRS, i_H)
        else:
            N_matched = 0

        if (N_H + N_WQRS - N_matched) != 0:
            b = int(N_matched*100 / (N_H + N_WQRS - N_matched))
        else:
            b = 0

        return b
    
    def __pasabajos__(self, signal, cut = 40):
        signal_ = signal.copy()
        order = 5
        b, a = scipy.signal.butter(order, cut, btype='lowpass', analog=False, fs= self.fs)
        signal1 = scipy.signal.filtfilt(b, a, signal_)
        return signal1

    def __pasaaltos__(self, signal):
        signal_ = signal.copy()
        cut = 0.7
        order = 2
        b, a = scipy.signal.butter(order, cut, btype='highpass', analog=False, fs=self.fs)
        signal1 = scipy.signal.filtfilt(b, a, signal_)
        return signal1

    def __extractor_pSQI__(self, signal):
        f, periodograma = scipy.signal.periodogram(signal, self.fs)
        try:
            i_5 = np.where(f > 5)[0][0]
            i_15 = np.where(f > 15)[0][0]
            i_40 = np.where(f > 40)[0][0]
            banda_1 = scipy.integrate.simps(periodograma[i_5:i_15], dx = self.fs)
            banda_2 = scipy.integrate.simps(periodograma[i_5:i_40], dx = self.fs)
            p = banda_1 / banda_2
        except:
            p = 0 
            print('error')
        return p
    
    def __extractor_basSQI__(self, signal):
        f, periodograma = scipy.signal.periodogram(signal, self.fs)
        i_1 = np.where(f > 1)[0][0]
        i_40 = np.where(f > 40)[0][0]
        banda_1 = scipy.integrate.simps(periodograma[0:i_1], dx = self.fs)
        banda_2 = scipy.integrate.simps(periodograma[0:i_40], dx = self.fs)
        bas = (1 - banda_1) / banda_2       # OJO! Varias versiones. Original: (1 - banda_1) / banda_2 . Dps: 1 - banda_1/banda_2
        return bas
    
    def __extractor_fSQI__(self, signal):
        temps = u.get_templates(signal, self.fs)
        med_temp = np.median(temps, axis = 0)
        amp_r = np.max(np.abs(med_temp))
        g_senal = np.gradient(signal)
        n = 0
        for g in g_senal:
            if g <= 0.01*amp_r:
                n = n + 1
        f = n*100/len(signal)
        return f

    def __extractor_carac__(self, window_bp, window_lp):
        pSQI = self.__extractor_pSQI__(window_lp)
        basSQI = self.__extractor_basSQI__(window_lp)
        fSQI = self.__extractor_fSQI__(window_lp)
        kSQI = scipy.stats.kurtosis(window_bp)
        sSQI = scipy.stats.skew(window_bp)
        X = [sSQI, kSQI, pSQI, basSQI, fSQI]
        return X