from pydantic import BaseModel, HttpUrl, ValidationError


class MyModel(BaseModel):
    url: HttpUrl


m = MyModel(url="http://www.example.com")
print()
print(type(str(m.url)))
