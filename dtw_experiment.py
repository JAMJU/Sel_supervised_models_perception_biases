#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    September 2019
    Juliette MILLET
    Code to experiment on dtw
"""
import dtw
import numpy.random as rd

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from scipy.spatial.distance import cdist
import random

def compute_dtw(x,y, dist_for_cdist, norm_div = False):
    """
    :param x:
    :param y: the two array must have the same number of columns (but the nb of lines can be different)
    :param dist_for_cdist: :param dist_for_cdist: distance used by cdist, can be 'braycurtis', 'canberra',
    'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    if 'kl', then instead of dtw with a distance, we use kl divergence
    :return: dtw distance between x and y
    """

    if dist_for_cdist == "kl":
        d = dtw.accelerated_dtw(x, y, dist = dtw.kl_divergence)[0]
    else:
        d = dtw.accelerated_dtw(x, y, dist = dist_for_cdist)[0]
    if not norm_div:
        return d
    else:
        return d/float(max(x.shape[0], y.shape[0]))

def compute_dtw_norm(x,y, dist_for_cdist, norm_div):
    """
        :param x:
        :param y: the two array must have the same number of columns (but the nb of lines can be different)
        :param dist_for_cdist: :param dist_for_cdist: distance used by cdist, can be 'braycurtis', 'canberra',
        'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
        if 'kl', then instead of dtw with a distance, we use kl divergence
        :return: dtw distance between x and y
    """

    if dist_for_cdist == "kl":
        D1, C, D2, path, norm, long = dtw.accelerated_dtw(x, y, dist=dtw.kl_divergence, norm_comput=True)
    else:
        D1, C, D2, path, norm, long = dtw.accelerated_dtw(x, y, dist=dist_for_cdist, norm_comput=True)

    #print('path', path)
    if not norm_div:
        return norm
    else:
        return float(norm)/float(long)

def have_all_dtwx(x,y, dist_for_cdist):
    """
    :param x:
    :param y: the two array must have the same number of columns (but the nb of lines can be different)
    :param dist_for_cdist: :param dist_for_cdist: distance used by cdist, can be 'braycurtis', 'canberra',
    'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    if 'kl', then instead of dtw with a distance, we use kl divergence
    :return: dtw distance between x and y
    """

    if dist_for_cdist == "kl":
        return dtw.accelerated_dtw(x, y, dist = dtw.kl_divergence)
    else:
        return dtw.accelerated_dtw(x, y, dist = dist_for_cdist)


def random_dtw(dimension, lenght_1, lenght_2, distance_for_cdist, nb_try):
    distance_obtained = []
    for i in range(nb_try):
        x = rd.ranf((lenght_1, dimension))
        y = rd.ranf((lenght_2, dimension))
        x = x/x.sum(axis = 1)[:, None]
        y = y/y.sum(axis = 1)[:, None]

        d = compute_dtw(x, y, dist_for_cdist=distance_for_cdist)
        distance_obtained.append(d)
    return np.mean(np.asarray(distance_obtained))




#def create_fake_posteriorgram_same_lenght(dimension, lenght, proba_pic):

#print(random_dtw(10, 5, 7, 'euclidean', 10))

def plot_evolution_with_dimension(length_1, lenght_2, distance_for_cdist, nb_try, min_dim, max_dim, name_fig):
    evol_dist = []
    for dim in range(min_dim, max_dim):
        evol_dist.append(random_dtw(dim, length_1, lenght_2, distance_for_cdist, nb_try))
    print(evol_dist)
    layout = go.Layout(
        title=go.layout.Title(
            text='Evol',
            font=dict(
                family='Times'
            ),
            xref='paper',
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='dimension',
                font=dict(
                    family='Times',
                    size=18,
                    color='#7f7f7f'
                )
            )

        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='random distance',
                font=dict(
                    family='Times',
                    size=18,
                    color='#7f7f7f'
                )
            ),


        )
    )
    fig = go.Figure(layout=layout)



    # you plot the correlation btwn x and y
    fig.add_scatter(
        x=list(range(min_dim, max_dim)),
        y=evol_dist,
        mode='lines',
        name='fitted line',
        showlegend=False
    )



    # save the figure
    pio.write_image(fig, name_fig + '.pdf')


#plot_evolution_with_dimension(length_1 = 10, lenght_2 = 13, distance_for_cdist = 'euclidean', nb_try = 15, min_dim = 2, max_dim=200, name_fig='euclidean')
#plot_evolution_with_dimension(length_1 = 10, lenght_2 = 13, distance_for_cdist = 'kl', nb_try = 15, min_dim = 2, max_dim=200, name_fig='kl')
#plot_evolution_with_dimension(length_1 = 10, lenght_2 = 13, distance_for_cdist = 'cosine', nb_try = 15, min_dim = 2, max_dim=200, name_fig='cosine')

def plot_evolution_with_len(length_min, lenght_max, distance_for_cdist, nb_try, dim, name_fig):
    evol_dist = []
    for leng in range(length_min, lenght_max):
        evol_dist.append(random_dtw(dim, leng, leng, distance_for_cdist, nb_try))
    #evol_dist = [0.9244184515300428, 1.3507927079775819, 1.8953683994705877, 2.4164033165307894, 3.078017969791574, 3.5451338898270066, 3.8834481672845627, 4.419339504421014, 4.929312150282599, 5.295734833813404, 5.677188914851658, 6.3604791638624345, 6.401814264195343, 7.04105355729374, 7.677628549582872, 8.167956434624223, 8.69908235446326, 8.995387920683337, 9.174546910183219, 10.08512368328054, 10.451475420024673, 10.636857194244064, 11.393734396557317, 11.98788877714612, 11.99260450811431, 12.777608772282964, 12.778001918199413, 13.423174160537071, 13.780544588074179, 14.392920760371453, 14.96848321383208, 15.46319106461876, 16.22948537989721, 16.425017514508678, 16.878242066406663, 17.17548436842818, 17.7774467130331, 18.20496954585564, 18.3381123333226, 19.255758106963853, 19.84398311128651, 19.84627547720434, 20.571929630668553, 20.910664435519838, 21.487493599042047, 21.78891178691278, 22.210448540716584, 23.359410022663074, 23.00198941136967, 23.390665909858498, 24.02559960143591, 24.867349020481875, 24.88244134695304, 25.98445817885347, 25.883857617614797, 26.406262437726692, 27.363585641640036, 27.62783716052433, 27.65636942183851, 28.36497986903198, 28.916510784921552, 29.13178885907545, 30.03558322310999, 30.024455252759488, 30.642342470720575, 30.87941129481068, 31.695468603126137, 31.555577077386122, 32.147240484857534, 33.14356474543625, 33.4930579623546, 33.845940385257016, 34.4251507329126, 34.6541119223024, 35.13020073716517, 35.435646100605105, 36.02059806027671, 35.878206059824414, 36.86396012833017, 37.139630699988516, 37.3972232842984, 38.064723046402975, 38.53765721451534, 39.238206941313805, 39.693134424957485, 39.85683533797666, 39.893693308593406, 40.99388383417763, 41.40841115956261, 41.80706268864515, 42.71695395325237, 42.36003200993532, 43.57041069241617, 43.72312624494503, 44.69736247639636, 44.3356386196287, 45.10948612403638, 45.87740286605979, 46.65415117978885, 46.42349349750527, 46.96884013373549, 46.986471459677155, 47.52170575028901, 48.63862899894237, 48.85977743480101, 49.42741050967709, 49.342940471237974, 49.71675807954058, 50.521044498722546, 51.41492390536246, 50.94726062485392, 51.764335272511666, 52.167366927752816, 52.481607295842885, 53.17333474877533, 53.4459672975531, 53.3550557453019, 54.92344608482172, 55.119760391444395, 55.12788907631439, 56.23660645259143, 56.95121330374064, 56.656807256110596, 56.963609698749856, 57.74024863755208, 58.581524222094906, 58.89197602684423, 59.102798593629046, 59.6928693318405, 60.24153671106531, 60.453148503910725, 61.1788992255016, 61.44209093999771, 61.664544443794355, 61.87036410811632, 62.63445407877545, 63.66630810846576, 63.79297960904965, 64.02874725539911, 65.05092869335913, 65.48925129176443, 65.28811453398272, 66.42282525343099, 66.60478814902703, 66.8905539933685, 67.44827964742701, 68.3705186658108, 67.95891234840168, 68.38101822826553, 68.91923283829, 69.29291644966138, 70.18665647244956, 69.91892372840867, 71.67342463090085, 71.40155550341318, 71.37516822196531, 71.89652606969257, 72.77351215753846, 72.38440462804617, 73.48218216041623, 74.57367645793732, 74.40755938022447, 75.61079871652382, 75.85369864185125, 75.91256417476934, 76.53561566996765, 77.08702184858045, 77.32665757110153, 78.16365920812619, 78.07940240422725, 78.89869906584177, 79.06192474679548, 79.89111660521982, 80.03806939455772, 80.40198602423798, 81.12453313119207, 81.08450539657146, 81.87680654261554, 82.16043702976306, 83.04337513323824, 82.91431388724827, 83.7666027701332, 83.58246288994455, 85.10454035814158, 85.22545986451196, 85.80602193488214, 85.80543781933584, 86.45845772078516, 86.66503187772597, 87.35958349187968, 87.21248332860984, 88.2721550004548, 88.52823547130133, 89.06569201993662, 89.28519705802732, 90.1514118047247, 90.75702366611661, 90.84336486934026]
    print(evol_dist)
    layout = go.Layout(
        title=go.layout.Title(
            text='Evol',
            font=dict(
                family='Times'
            ),
            xref='paper',
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='lenght x and y',
                font=dict(
                    family='Times',
                    size=18,
                    color='#7f7f7f'
                )
            )

        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='random distance',
                font=dict(
                    family='Times',
                    size=18,
                    color='#7f7f7f'
                )
            ),


        )
    )
    fig = go.Figure(layout=layout)



    # you plot the correlation btwn x and y
    fig.add_scatter(
        x=list(range(length_min, lenght_max)),
        y=evol_dist,
        mode='lines',
        name='fitted line',
        showlegend=False
    )



    # save the figure
    pio.write_image(fig, name_fig + '.pdf')

def plot_dtw(x, y, dist):

    scale = 3/x.shape[1]

    d, C, D1, path, norm= have_all_dtwx(x, y, dist_for_cdist=dist)
    print('path', len(path))
    print('the distance found is ', d)
    if dist != 'kl':
        D0 = cdist(x, y, dist)
    else:
        D0 = cdist(x, y, dtw.kl_divergence)


    heatmap = go.Heatmap(z=D0,coloraxis = "coloraxis2", dx =1, dy = 1)

    layout = go.Layout(
        coloraxis1={"colorscale":"Viridis", "colorbar":{"title":"data to compare", "x":1.08}},
        coloraxis2={ "colorbar":{"title":"path value"}}
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Heatmap(z=y.swapaxes(0,1)))
    fig2.show()

    fig3 = go.Figure()
    fig3.add_trace(go.Heatmap(z=x.swapaxes(0,1)))
    fig3.show()

    fig = go.Figure(layout = layout)
    fig.add_scatter(x=path[1],
        y=path[0],
        name='optimal path',
        showlegend=False
    )

    fig.add_trace(go.Heatmap(z = y.swapaxes(0,1), coloraxis = "coloraxis1",x0 = 0, y0 =D0.shape[0] + 1, dy = scale, dx = 1))
    fig.add_trace(go.Heatmap(z = x, coloraxis = "coloraxis1",  x0=D0.shape[1] + 1, y0 = 0, dx = scale, dy = 1))
    # save the figure
    #pio.write_image(fig, 'test' + '.pdf')

    # Add Heatmap Data to Figure

    fig.add_trace(heatmap)

    pio.write_image(fig, 'test' + '.pdf')

    # Plot!
    fig.show()
    #plot(x=D1.shape[0], y = D1.shape[1], z = t(m), type = "contour")
    #add_trace(x=c(1, 2, 3, 4, 5, 6), y=c(1, 2, 3, 3, 4, 5), type="scatter", mode="line")


def create_fake_posterior_gram(same, randomi, long, mult_compare_to_noise, lenght_1, lenght_2, dimension):
    if lenght_1 > lenght_2:
        lenght_1, lenght_2 = lenght_2, lenght_1

    if same:
        to_activate = np.random.randint(dimension, size=lenght_1)
        x = rd.ranf((lenght_1, dimension))
        y = rd.ranf((lenght_1, dimension))

        for i in range(lenght_1):
            x[i, to_activate[i]] = mult_compare_to_noise
            y[i, to_activate[i]] = mult_compare_to_noise

        x = x / x.sum(axis=1)[:, None]
        y = y / y.sum(axis=1)[:, None]
        return x, y
    if randomi:
        to_activate_1 = np.random.randint(dimension, size=lenght_1)
        to_activate_2 = np.random.randint(dimension, size=lenght_2)
        x = rd.ranf((lenght_1, dimension))
        y = rd.ranf((lenght_2, dimension))

        for i in range(lenght_1):
            x[i, to_activate_1[i]] = mult_compare_to_noise
        for i in range(lenght_2):
            y[i, to_activate_2[i]] = mult_compare_to_noise

        x = x / x.sum(axis=1)[:, None]
        y = y / y.sum(axis=1)[:, None]
        return x, y
    if long:
        to_activate = np.random.randint(dimension, size=lenght_1)
        x = rd.ranf((lenght_1, dimension))
        y = rd.ranf((lenght_2, dimension))
        for i in range(lenght_1):
            x[i, to_activate[i]] = mult_compare_to_noise
        iter = 0
        to_copy = 0
        to_double = [random.randint(0,lenght_1 - 1) for i in range(lenght_2 - lenght_1)]
        to_double = sorted(to_double)
        begin = 0
        while iter < lenght_2:
            if to_copy in to_double[begin:]:
                y[iter, :] = x[to_copy, :]
                begin += 1
                iter += 1
            else:
                y[iter, :] = x[to_copy, :]
                to_copy += 1
                iter += 1


        x = x / x.sum(axis=1)[:, None]
        y = y / y.sum(axis=1)[:, None]
        return x, y


def evolution_distance_different_lenght_but_same(lenght_to_compare, lenght_min, lenght_max, dimension, distance, nb_try, name_fig, compare_with_mean_mat, norm = False):
    value = [0 for i in range(lenght_min, lenght_max)]
    mean_mat = [0 for i in range(lenght_min, lenght_max)]
    for l in range(lenght_min, lenght_max):
        for i in range(nb_try):
            x, y = create_fake_posterior_gram(same=False, randomi=True, long=False, mult_compare_to_noise=100, lenght_1=lenght_to_compare,
                                              lenght_2=l, dimension=dimension)
            if compare_with_mean_mat:
                if distance != 'kl':
                    mean_mat[l - lenght_min] += cdist(x,y, distance).mean()
                else:
                    mean_mat[l-lenght_min] += cdist(x,y, dtw.kl_divergence).mean()
            if not norm:
                value[l - lenght_min] += compute_dtw(x, y, distance)
            else:
                value[l - lenght_min] += compute_dtw_norm(x, y, distance)

        value[int(l - lenght_min)]  = float(value[l - lenght_min]) / float(nb_try)
        #mean_mat[int(l - lenght_min)] = float(mean_mat[l - lenght_min]) / float(nb_try)

    layout = go.Layout(
        title=go.layout.Title(
            text='Evol',
            font=dict(
                family='Times'
            ),
            xref='paper',
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='dimension',
                font=dict(
                    family='Times',
                    size=18,
                    color='#7f7f7f'
                )
            )

        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='distance between different lenght',
                font=dict(
                    family='Times',
                    size=18,
                    color='#7f7f7f'
                )
            ),

        )
    )
    fig = go.Figure(layout=layout)

    # you plot the correlation btwn x and y
    fig.add_scatter(
        x=list(range(lenght_min, lenght_max)),
        y=value,
        mode='lines',
        name='dtw distance',
        showlegend=True
    )
    if compare_with_mean_mat:
        # you plot the correlation btwn x and y
        fig.add_scatter(
            x=list(range(lenght_min, lenght_max)),
            y=mean_mat,
            mode='lines',
            name='mean of all distances',
            showlegend=True
        )
    # save the figure
    pio.write_image(fig, name_fig + '.pdf')

def create_posteriorgram(wide, begin_nb, middle_nb, end_nb, id_beg, id_mid, id_end):
    post = np.zeros((begin_nb + middle_nb + end_nb, wide))

    for i in range(begin_nb):
        post[i, id_beg] = 1.
    for i in range(middle_nb):
        post[i + begin_nb, id_mid] = 1.
    for i in range(end_nb):
        post[i + begin_nb + middle_nb, id_end] = 1.
    return np.asarray(post, dtype = float)


def add_noise(post, amount_noise):
    noise = np.abs(np.random.normal(0, amount_noise, post.shape))
    #print(noise)
    total = post + noise
    result = np.zeros(total.shape)
    for i in range(total.shape[0]):
        sum = np.sum(total[i])
        for j in range(total.shape[1]):
            result[i,j] = total[i,j]/sum
    return result

def plot_ABX_evolution( wide, id_TGT, id_OTH, begin_mid, end_mid, noise_X = False, norm = False, noise_TGT_OTH = False, norm_div = False):
    TGT = create_posteriorgram(wide=wide, begin_nb=10, middle_nb=20, end_nb=10, id_beg=1, id_mid=id_TGT, id_end=5)
    OTH = create_posteriorgram(wide=wide, begin_nb=10, middle_nb=20, end_nb=10, id_beg=1, id_mid=id_OTH, id_end=5)

    value_TGT = []
    value_OTH = []
    diff = []

    if noise_TGT_OTH:
        TGT = add_noise(TGT, 0.01)
        OTH = add_noise(OTH, 0.01)
    for i in range(begin_mid, end_mid):
        X = create_posteriorgram(wide=wide, begin_nb=10, end_nb=10, middle_nb=i,  id_beg = 1, id_mid=id_TGT,  id_end = 5)
        #print(X.shape)
        if noise_X:
            X = add_noise(X, 0.01)

        if not norm:
            TGT_X = compute_dtw(X, TGT, dist_for_cdist= 'kl',norm_div = norm_div)
            OTH_X = compute_dtw(X, OTH, dist_for_cdist='kl', norm_div = norm_div)

        else:
            TGT_X = compute_dtw_norm(X, TGT, dist_for_cdist='kl', norm_div=norm_div)
            OTH_X = compute_dtw_norm(X, OTH, dist_for_cdist='kl', norm_div=norm_div)
        print(OTH_X)
        value_TGT.append(TGT_X)
        value_OTH.append(OTH_X)
        diff.append(OTH_X - TGT_X)

    layout = go.Layout(
        title=go.layout.Title(
            text='Evolution of the DTW distances, noise = 0.01 (using kl-div)',
            font=dict(
                family='Times'
            ),
            xref='paper',
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='lenght change of X',
                font=dict(
                    family='Times',
                    size=18,
                    color='#7f7f7f'
                )
            )

        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='distance',
                font=dict(
                    family='Times',
                    size=18,
                    color='#7f7f7f'
                )
            ),

        )
    )
    fig = go.Figure(layout=layout)

    # you plot the correlation btwn x and y
    fig.add_scatter(
        x=list(range(begin_mid, end_mid)),
        y=value_TGT,
        mode='lines',
        name='with TGT',
        showlegend=True
    )
    fig.add_scatter(
        x=list(range(begin_mid, end_mid)),
        y=value_OTH,
        mode='lines',
        name='with OTH',
        showlegend=True
    )

    pio.write_image(fig, "diff_TGT_OTH" + '.png')

    fig = go.Figure(layout=layout)
    fig.add_scatter(
        x=list(range(begin_mid, end_mid)),
        y=diff,
        mode='lines',
        name='delta',
        showlegend=True
    )
    #fig.show()
    pio.write_image(fig, "diff_delta" + '.png')
#TGT = create_posteriorgram(wide=10, begin_nb=10, middle_nb=20, end_nb=10, id_beg=1, id_mid=3, id_end=5)
#OTH = create_posteriorgram( wide=10, begin_nb=10, middle_nb=20, end_nb=10, id_beg=1, id_mid=4, id_end=5)
#plot_dtw(x = TGT, y = OTH, dist = "kl")
#A = np.asarray([[0.1,1], [0.1,1], [0.1,1]])
#B = np.asarray([[0.1,1],[0.1,0.1], [0.1,2], [0.1,3]])
#from scipy.spatial.distance import cdist
#C = cdist(A,B, 'cosine')
if __name__ == '__main__':
    plot_ABX_evolution(wide = 600, begin_mid=5, end_mid=70, id_OTH=3, id_TGT=4, noise_X=True, noise_TGT_OTH=False, norm = True, norm_div=True)




#x, y = create_fake_posterior_gram(same=False, randomi=False, long = True, mult_compare_to_noise=15, lenght_1=30, lenght_2=100, dimension=600)
#d, C, D1, path, norm = dtw.accelerated_dtw(x, y, dist = 'cosine', norm_comput=True)
#print(d, norm)
#print(x.shape)
#print(y.shape)
#plot_dtw(x = x, y = y, dist = 'cosine')
# A FAIRE:
# Visualisation du matching (avec dessin des representation sur les cotés
#
#
#evolution_distance_different_lenght_but_same(lenght_to_compare=30, lenght_min=15, lenght_max=100, dimension=100,
#                                             distance='cosine', nb_try=10, name_fig='from_15_to_200_kl_dim100_with_norm_random',
#                                             compare_with_mean_mat=False, norm=True)