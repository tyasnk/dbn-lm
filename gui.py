from tkinter import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1)).fit(train)

lag = 3
fields = []
for i in range(lag):
    if (i + 1) - lag != 0:
        fields.append(
            "Masukkan nilai tukar %.0f hari sebelumnya : " % (lag - (i + 1)))
    else:
        fields.append("Masukkan nilai tukar hari ini : ")


def fetch(entries):
    input_form = []
    for entry in entries:
        text = entry[1].get()
        input_form.append(text)
    input_pred = np.array(input_form)
    input_pred = scaler.transform(input_pred)
    input_pred = np.reshape(input_pred, (1, lag, 1))
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=save_path)
        out_pred = sess.run(pred, feed_dict={Z: input_pred})
        out_pred = np.array(out_pred)
    out_pred = np.reshape(out_pred, (1, -1))
    out_pred_transform = int(scaler.inverse_transform(out_pred))
    textbox.insert(END,
                   'Prediksi nilai tukar mata uang esok hari : %i \nTingkat Keyakinan Akurasi : %.4f%%' %
                   (out_pred_transform, Dstat_onTest))
    return print(input_pred)


def makeform(root, fields):
    entries = []
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=32, text=field, anchor='w')
        ent = Entry(row)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append((field, ent))
    return entries


if __name__ == '__main__':
    root = Tk()
    root.wm_title(
        'model GRU-RNN untuk peramalan nilai tukar mata uang Rupiah terhadap Dolar Amerika')
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))
    b1 = Button(root, text='Prediksi',
                command=(lambda e=ents: fetch(e)))
    b1.pack(side=LEFT, padx=5, pady=5)
    textbox = Text(root, height=3)
    textbox.pack(side=RIGHT, padx=5, pady=5)
    root.mainloop()