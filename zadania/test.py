

class Klasa:
    klasowa = 'klasowa1'

    def __init__(self):
        print ('---------------init')
        initowa = 'initowa1'                # tylko w init
        self.selfinitowa = 'selfinitowa1'

    def fun(swoj):
        print('----------------fun')
        # print(initowa)            # tu nie ma
        # print(klasowa)             # to nie jest klasowa
        print(swoj.klasowa)
        print(swoj.selfinitowa)
        klasowa = 'klasowa2'        # tylko lokalna
        print(swoj.klasowa)         # klasowa
        print(klasowa)              # lokalna
        swoj.klasowa = 'klasowa3'


globalna = 'globalna1'
globalna2 = 'globalna21'


def test1():
    print('Hi')
    print(globalna)
    print(globalna2)
# test1()


skrot = Klasa           # skrót do Klasa
k = skrot()             # utworzenie obiektu


def test2():
    print(skrot)            # skrot __main__.Klasa
    print(k)                # obiekt
    print(globalna)
    print(globalna2)
    # print(klasowa)        # nie ma
    print(Klasa.klasowa)    # przez klasę
    print(k.klasowa)        # przez obiekt
    # print(initowa)        # global nie ma
    # print(k.initowa)      # obiekt też nie ma
    # print(Klasa.selfinitowa)   # klasa nie ma
    print(k.selfinitowa)        #obiekt ma
# test2()


# k.fun()


def test3():
    Klasa.klasowa = 'klasowa4'
    print(Klasa.klasowa)
    k2 = Klasa()
    print(k2.klasowa)
    k2.klasowa = 'klasowa5'
    print(Klasa.klasowa)
    print(k2.klasowa)
    Klasa.klasowa = 'klasowa6'
    print(k.klasowa)
# test3()

