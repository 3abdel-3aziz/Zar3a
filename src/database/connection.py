import sys
import os

from  sqlalchemy import create_engine 
from sqlalchemy.orm import declarative_base , sessionmaker 


# pyrefly: ignore [missing-import]
from config import settings 


if settings.is_sqlite:
    engine= create_engine(

        settings.DATABASE_URL, connect_args= {"check_same_thread" :False}
    )
else :
    engine = create_engine(settings.DATABASE_URL)    


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():

    db= SessionLocal()
    try:
        yield db
    finally:
        db.close()
