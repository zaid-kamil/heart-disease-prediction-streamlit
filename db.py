import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# setup db code
Base = declarative_base()

# create table as python class
class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer,primary_key=True)
    patient = Column(String)
    features = Column(String)
    result = Column(Integer)
    uploaded_on = Column(DateTime,default=datetime.now)

    def __str__(self):
        return f"{self.id} : {self.patient}"
    
    def __repr__(self) -> str:
        return f"{self.id} : {self.patient}"

# create database
if __name__ == "__main__":
    engine = create_engine("sqlite:///db.sqlite3")
    Base.metadata.create_all(engine)