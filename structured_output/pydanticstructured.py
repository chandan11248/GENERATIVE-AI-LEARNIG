from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class student(BaseModel):
    name:str
    age:Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0,lt=10,default=2,description="representation of cgpa ")

new_student=student(name="chandan",age=20,email="chandan@gmail.com")
print(new_student)
STU_dict=dict(new_student)
print(STU_dict)