from collections import OrderedDict
import cv2
import sys
import math
from mreza import neuronska_mreza as nm
import numpy as np

class Pratilac:

    def __init__ (self, plava_linija, zelena_linija):
        self.id_objekta = 0
        self.pronadjeni_objekti = OrderedDict()
        self.izgubljeni_objekti = OrderedDict()
        self.stari_objekti = OrderedDict()
        self.plava_linija = plava_linija
        self.zelena_linija = zelena_linija
        self.kadar = 0
        self.suma = 0

    def obradi(self, contours, frame) :
        self.kadar += 1

        pronadjeni_u_frejmu = OrderedDict()

        for contour in contours:
            momenti = cv2.moments(contour)
            povrsina = momenti["m00"]

            try:
                centroid = (momenti["m10"]/momenti["m00"], momenti["m01"]/momenti["m00"]) #centar konture, moze se uslovno posmatrati kao pozicija
            except ZeroDivisionError: # greska se moze javiti ukoliko je nulti moment jednak nuli, odnosno ako kontura nema nijedan beo piksel, do cega ne bi trebalo da dodje 
                continue

            ivica = 15

            height, width, _ = frame.shape
            
            if centroid[0] > (width - ivica):
                continue
            if centroid[1] > (height - ivica):
                continue

            # pretraga se prvo vrsi u recniku objekata koji su pronadjeni u prethodnom kadru 
            if len(self.pronadjeni_objekti):
                objekti = self.pronadji_u_blizini(centroid, self.pronadjeni_objekti, 5.0)
                if len(objekti):
                    objekat = self.izaberi_po_povrsini(objekti, povrsina)

                    if self.kadar % 12 == 0:
                        self.izvrsi_predikciju(objekat, contour, frame)


                    # if math.hypot(objekat.trenutna_lokacija[0]-self.plava_linija[0][0], objekat.trenutna_lokacija[1]-self.plava_linija[0][1]) <= 3:
                    #   objekat.presao_plavu = True
                    
                    # if math.hypot(objekat.trenutna_lokacija[0]-self.plava_linija[1][0], objekat.trenutna_lokacija[1]-self.plava_linija[1][1]) <= 3:
                    #   objekat.presao_plavu = True

                    # if math.hypot(objekat.trenutna_lokacija[0]-self.zelena_linija[0][0], objekat.trenutna_lokacija[1]-self.zelena_linija[0][1]) <= 3:
                    #   objekat.presao_zelenu = True
                    
                    # if math.hypot(objekat.trenutna_lokacija[0]-self.zelena_linija[1][0], objekat.trenutna_lokacija[1]-self.zelena_linija[1][1]) <= 3:
                    #   objekat.presao_zelenu = True

                    # if da_li_se_seku(objekat.prvi_pronalazak, objekat.trenutna_lokacija, self.plava_linija[0], self.plava_linija[1]):
                    #     objekat.presao_plavu = True
                    
                    # if da_li_se_seku(objekat.prvi_pronalazak, objekat.trenutna_lokacija, self.zelena_linija[0], self.zelena_linija[1]):
                    #     objekat.presao_zelenu = True
                    

                    objekat.trenutna_lokacija = centroid
                    pronadjeni_u_frejmu[objekat.id] = objekat
                    del self.pronadjeni_objekti[objekat.id]
                    continue
                
            # pretraga se vrsi u objektima koji su nisu pronadjeni u prethodnom kadru
            if len(self.izgubljeni_objekti):
                objekti = self.pronadji_u_blizini(centroid, self.izgubljeni_objekti, 50.0)
                if (len(objekti)):
                    #print('stari')
                    objekat = self.izaberi_po_povrsini(objekti, povrsina)

                    self.izvrsi_predikciju(objekat, contour, frame)

                    if da_li_se_seku(objekat.prvi_pronalazak, objekat.trenutna_lokacija, self.plava_linija[0], self.plava_linija[1]):
                        objekat.presao_plavu = True
                    
                    if da_li_se_seku(objekat.prvi_pronalazak, objekat.trenutna_lokacija, self.zelena_linija[0], self.zelena_linija[1]):
                        objekat.presao_zelenu = True

                    objekat.trenutna_lokacija = centroid
                    objekat.izgubljen = 0
                    pronadjeni_u_frejmu[objekat.id] = objekat
                    del self.izgubljeni_objekti[objekat.id]
                    continue

            # odnosi se na objekte koji su se po prvi put pojavili u video zapisu
            # a pokriva i one objekte koji su duzi period izgubljeni
            objekat = Praceni(self.id_objekta, centroid, povrsina)
            pronadjeni_u_frejmu[self.id_objekta] = objekat

            self.izvrsi_predikciju(objekat, contour, frame)

            self.id_objekta += 1

        # vrsi se nakon prolaska kroz sve identifikovane konture
        self.izgubljeni_objekti.update(self.pronadjeni_objekti)
        ids = []

        for obj in self.izgubljeni_objekti.items():
            obj[1].izgubljen += 1
            ids.append(obj[1].id)
        
        # uklanjanje objekata koji su duze od 50 kadrova izgubljeni
        # vrsi se i provera eventualnog prelaska neke od linija,
        # i informacija o tome se cuva u recniku izgubljenih objekata
        for id in ids:

            if (self.izgubljeni_objekti[id].izgubljen > 50):
                obj = self.izgubljeni_objekti[id]
                prvi_pronalazak = obj.prvi_pronalazak
                trenutna_lokacija = obj.trenutna_lokacija
                
                if da_li_se_seku(prvi_pronalazak, trenutna_lokacija, self.plava_linija[0], self.plava_linija[1]):
                    obj.presao_plavu = True
                    self.stari_objekti[id] = obj
                
                if da_li_se_seku(prvi_pronalazak, trenutna_lokacija, self.zelena_linija[0], self.zelena_linija[1]):
                    obj.presao_zelenu = True
                    self.stari_objekti[id] = obj


                del self.izgubljeni_objekti[id]

        self.pronadjeni_objekti = pronadjeni_u_frejmu

        self.izasli_iz_kadra(self.pronadjeni_objekti, 10.0)
        self.izasli_iz_kadra(self.izgubljeni_objekti, 20.0)

        # print(self.stari_objekti, len(self.stari_objekti))
        # print('ukupno ih ima ', len(self.pronadjeni_objekti), ' izgubljenih: ', len(self.izgubljeni_objekti), ' izaslih: ', len(self.stari_objekti))
        # self.pronadjeni_objekti = pronadjeni_u_frejmu
        # print(len(pronadjeni_u_frejmu), ' ukupno')

    def izvrsi_predikciju(self, objekat, contour, frame):
        x,y,w,h = cv2.boundingRect(contour)
        img = frame[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.resize(img, (28, 28))

        # a = img.copy()

        img = np.asarray(img, dtype="int32")

        img = np.reshape(img, (1,784))

        p = nm.predict(img)
        #print(p, np.argmax(p))

        objekat.vrednost += p

        # print(objekat.id, ': ', objekat.vrednost, np.argmax(objekat.vrednost), objekat.presao_plavu, objekat.presao_zelenu)

        # cv2.imshow('aaa', a)

        # if cv2.waitKey(0) & 0xFF == ord('n'):
        #     pass

    def izracunaj(self):
        presli_plavu = 0
        for o in self.pronadjeni_objekti.values():            
            if da_li_se_seku(o.prvi_pronalazak, o.trenutna_lokacija, self.plava_linija[0], self.plava_linija[1]):
                o.presao_plavu = True

            if da_li_se_seku(o.prvi_pronalazak, o.trenutna_lokacija, self.zelena_linija[0], self.zelena_linija[1]):
                o.presao_zelenu = True

            if o.presao_plavu:
                presli_plavu += 1
                self.suma += np.argmax(o.vrednost)

            if o.presao_zelenu:
                self.suma -= np.argmax(o.vrednost)

        for o in self.stari_objekti.values():
            
            if da_li_se_seku(o.prvi_pronalazak, o.trenutna_lokacija, self.plava_linija[0], self.plava_linija[1]):
                o.presao_plavu = True

            if da_li_se_seku(o.prvi_pronalazak, o.trenutna_lokacija, self.zelena_linija[0], self.zelena_linija[1]):
                o.presao_zelenu = True

            if o.presao_plavu:
                presli_plavu += 1
                self.suma += np.argmax(o.vrednost)

            if o.presao_zelenu:
                self.suma -= np.argmax(o.vrednost)

        # print('presli plavu ', presli_plavu)
        return self.suma

    def izasli_iz_kadra(self, objects, area):
        izasli = []
        for o in objects.items():
            if o[1].trenutna_lokacija[0] + area >= 640:
                izasli.append(o[1].id)

            elif o[1].trenutna_lokacija[1] + area >= 480:
                izasli.append(o[1].id)
        
        for id in izasli:
            self.stari_objekti[id] = objects[id]
            del objects[id]
            
    def izaberi_po_povrsini(self, objects, area):
        najpriblizniji = (objects[0], abs(objects[0].velicina - area))

        for o in objects:
            if abs(o.velicina-area) < najpriblizniji[1]:
                najpriblizniji = (o, abs(o.velicina-area))

        return najpriblizniji[0]

    def pronadji_u_blizini(self, centroid, objects, radius):
        ret_lista = []

        for o in objects.values():
            if math.hypot(o.trenutna_lokacija[0]-centroid[0], o.trenutna_lokacija[1]-centroid[1]) <= radius:
                ret_lista.append(o)

        return ret_lista 
      

class Praceni: 

    def __init__(self, id, location, velicina):
        self.id = id
        self.prvi_pronalazak = location
        self.trenutna_lokacija = location
        self.presao_plavu = False
        self.presao_zelenu = False
        self.velicina = velicina
        self.izgubljen = 0
        self.vrednost = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def da_li_se_seku(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)