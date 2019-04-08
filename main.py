import numpy as np
import cv2
import os
import util
import pronalazenje_linije as pl
import pronalazenje_kontura as pk
import pracenje as p

def read_frame(file, output_file) :
    path=os.path.join("dataset", file)
    cap = cv2.VideoCapture(path)

    _, frame = cap.read()
    
    plava_linija = (x1, y1), (x2, y2) = pl.pronadji_liniju(util.BLUE, frame.copy())
    zelena_linija = (x1, y1), (x2, y2) = pl.pronadji_liniju(util.GREEN, frame.copy())

    pratilac = p.Pratilac(plava_linija, zelena_linija)

    while(True):

        if frame is None:
            break

        cv2.line(frame, plava_linija[0], plava_linija[1], (0,0,255), 2)
        cv2.line(frame, zelena_linija[0], zelena_linija[1], (0,0,255), 2)

        contours = pk.pronadji_konture(frame.copy())

        pratilac.obradi(contours, frame.copy())

        # cv2.imshow(file,frame)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        _, frame = cap.read()

    suma = pratilac.izracunaj()
    output_file.write(file + '\t' + str(suma) + '\n')

    cap.release()
    cv2.destroyAllWindows()

output_file = open("out.txt","w")
output_file.write('RA 200/2015 Marko Vukovic\n')
output_file.write('file\tsum\n')

for file in os.listdir("dataset"):
    if file.endswith(".avi"):
        print('obradjuje se: ', file)
        read_frame(file, output_file)
        # break # da prikaze samo jedan fajl
output_file.close()
