from pydantic import BaseModel,EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name : str = 'dishita' # setting default value
    age : Optional[int] = None 
    email : EmailStr
    cgpa : float = Field(gt=0 , lt=10 , default = 5, description='A decimal value representing the cgpa of the student')

new_student = {'age' : '21' , 'email':'abc@gmail.com', } # here int should be passed but str in passed but pydantic is smart enough to analyze that there is a number in the format of str so it will extract the number from it.

# object of Student class
student = Student(**new_student)

print(student)

student_dict = dict(student)
print(student_dict['age'])

student_json = student.model_dump_json()
print(student_json)
