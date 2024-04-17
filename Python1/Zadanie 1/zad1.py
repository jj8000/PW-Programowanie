class Pracownik:
    def __init__(self, imie, nazwisko, wyn, stan):
        self.imie = imie
        self.nazwisko = nazwisko
        self.stan = stan
        self._wyn = wyn
        self.urlop = 26
        
    def check(self, obj):
        print(f"Imię: {obj.imie}")
        print(f"Nazwisko: {obj.nazwisko}")
        print(f"Stanowskio: {obj.stan}")
        
    def wez_urlop(self, d):
        if d>self.urlop:
            print(f"Masz do dyspozycji tylko {self.urlop} dni urlopu!")
        else:   
            self.urlop-=d
            print(f"Zaplanowano {d} dni urlopu")
            print(f"Pozostało {self.urlop} dni urlopu")
    
    def work(self):
        raise Exception()
    
        
class Brygadzista(Pracownik):
    def __init__(self, imie, nazwisko, wyn, stan="Brygadzista"):
        super().__init__(imie, nazwisko, wyn, stan)
    
    def work(self):
        print("Pracuję jako brygadzista!")
  
        
class Specjalista(Pracownik):
    def __init__(self, imie, nazwisko, wyn, stan="Specjalista"):
        super().__init__(imie, nazwisko, wyn, stan)
        
    def work(self):
        print("Pracuję jako specjalista!")
  
        
class Technik(Pracownik):
    def __init__(self, imie, nazwisko, wyn, stan="Technik"):
        super().__init__(imie, nazwisko, wyn, stan)
        
    def work(self):
        print("Pracuję jako technik!")

        
class Kierownik(Pracownik):
    def __init__(self, imie, nazwisko, wyn, stan="Kierownik"):
        super().__init__(imie, nazwisko, wyn, stan)
        
    def work(self):
        print("Kieruję wszystkim!")
    
    def zmiana_wyn(self, obj, num):
        if obj==self:
            print("Nie możesz dać podwyżki samemu sobie!")
        else:
            obj._wyn+=num
            print(f"{obj.imie} {obj.nazwisko}: nowe wynagrodzenie wynosi {obj._wyn}")
    
        
#Testy
b1 = Brygadzista("Stefan", "Kowalski", 10000)
s1 = Specjalista("Jan", "Nowak", 8000)
t1 = Technik("Zbigniew", "Malinowski", 7000)
t2 = Technik("Anna", "Malinowska", 8000)
k = Kierownik("Adam", "Kwiatkowski", 20000)

b1.check(s1)
k.zmiana_wyn(k, 10000)

s1.wez_urlop(10)
s1.wez_urlop(30)

b1.work()
s1.work()
t1.work()
t2.work()
k.work()