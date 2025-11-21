

class associate(object):

    def __init__(self, asso_id, asso_name):
        self.id = asso_id
        self.name = asso_name
    
    def display_info(self):
        print(self.id)
        print(self.name)

class employee(associate):
    
    def __init__(self, asso_id, asso_name, emp_dept, emp_salary):
        self.dept = emp_dept
        self.salary = emp_salary
        associate.__init__(self, asso_id, asso_name)

    def emp_info(self):
        print(self.id)
        print(self.name)
        print(self.dept)
        print(self.salary)

a = employee(1, "c k", "IT",20000.43)

a.emp_info()




class Bird: 
    
    def intro(self): 
        print("There are many types of birds.") 
  
    def flight(self): 
        print("Most of the birds can fly but some cannot.") 
  
class sparrow(Bird): 
    
    def flight(self): 
        print("Sparrows can fly.") 
  
class ostrich(Bird): 
  
    def flight(self): 
        print("Ostriches cannot fly.") 
  
obj_bird = Bird() 
obj_spr = sparrow() 
obj_ost = ostrich() 
  
obj_bird.intro() 
obj_bird.flight() 
  
obj_spr.intro() 
obj_spr.flight() 
  
obj_ost.intro() 
obj_ost.flight() 



