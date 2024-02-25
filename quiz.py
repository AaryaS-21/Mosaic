import math
import random

class Quiz:
    def __init__(self, res):
        self.results = res
        self.strength = 50 #
        self.tempo = 50 #
        self.social = 50 #
        self.priority = 50 #
        self.style = 50
        self.flexibility = 50
        self.competitive = 50 #
        self.values = 50
        self.environment = 50 #
        self.tech = 50 #
        self.attributes = [self.strength, self.tempo, self.social, self.priority, self.style, self.flexibility, self.tech]
        print(self.calculate_scores())
        self.create_personality()
    
    def generate_personality(self):
        self.results = []
        for i in range(0, 40):
            self.results.append(random.randint(1, 5))
    
    def create_personality(self):
        self.results = self.calculate_scores()
        careers = ""
        tech = False
        environment = False
        if self.tech > 0.5:
            tech = True
        if self.environment > 0.5:
            environment = True
        if self.priority < 0.3:
            careers = "Humanitarian"
        elif self.strength > 0.75:
            if self.flexibility > 0.6:
                careers = "Problem Solver"
            else:
                careers = "Analyst"
        elif self.strength < 0.2:
            careers = "Dreamer"
        elif self.flexibility > 0.7 and self.tempo > 0.7 and self.social > 0.5:
            careers = "Maverick"
        elif self.competitive > 0.75 and self.social > 0.6 and self.style > 0.5:
            careers = "Advocate"
        elif self.values < 0.5 and self.style > 0.5 and self.flexibility > 0.5:
            careers = "Go-Getter"
        else:
            careers = 'Dreamer'
        print(careers)
    
    def calculate_scores(self):
        for i in range(0, 39):
            print(self.results)
            result = self.results[i]-3
            if i == 0:
                self.competitive += (result*10)
            elif i == 1:
                self.priority -= (result*5)
            elif i == 2:
                self.social -= (result*7)
            elif i == 3:
                self.tempo += (result*5)
            elif i == 4:
                self.social += (result*7)
            elif i == 5:
                self.environment += (result*5)
            elif i == 6:
                self.tempo += (result*5)
            elif i == 7:
                self.strength -= (result*5)
            elif i == 8:
                self.priority += (result*3)
                self.strength += (result*5)
            elif i == 9:
                self.values -= (result*12)
            elif i == 10:
                self.social -= (result*7)
            elif i == 11:
                self.social += (result*7)
                self.style += (result*7)
            elif i == 12:
                self.tempo += (result * 5)
            elif i == 13:
                self.tempo -= (result*5)
            elif i == 14:
                self.strength -= (result*5)
            elif i == 15:
                self.social += (result*7)
                self.competitive += (result * 5)
            elif i == 16:
                self.style -= (result*7)
            elif i == 17:
                self.priority += (result*5)
            elif i == 18:
                self.flexibility -= (result*4)
            elif i == 19:
                self.social += (result*7)
            elif i == 20:
                self.priority -= (result*5)
            elif i == 21:
                self.priority += (result*5)
            elif i == 22:
                self.environment -= (result*10)
            elif i == 23:
                self.social += (result*7)
            elif i == 24:
                self.flexibility += (result*4)
                self.style += (result * 7)
            elif i == 25:
                self.style -= (result * 7)
            elif i == 26:
                self.strength += (result*5)
                self.tech += (result*25)
            elif i == 27:
                self.style += (result * 7)
            elif i == 28:
                self.style -= (result * 7)
            elif i == 29:
                self.tempo -= (result*5)
            elif i == 30:
                self.flexibility += (result * 4)
            elif i == 31:
                self.style += (result * 7)
            elif i == 32:
                self.strength -= (result * 5)
            elif i == 33:
                self.values -= (result*12)
            elif i == 34:
                self.competitive += (result*5)
            elif i == 35:
                self.flexibility += (result*4)
            elif i == 36:
                self.flexibility += (result*4)
            elif i == 37:
                self.competitive -= (result*5)
            elif i == 38:
                self.flexibility -= (result*4)
            elif i == 39:
                self.environment += (result*10)
        self.attributes = [self.strength, self.tempo, self.social, self.priority, self.style, self.flexibility, self.tech]
        return self.attributes

