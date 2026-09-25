from datetime import datetime 
from sqlalchemy import Column , DateTime ,Boolean, ForeignKey , Integer , String , Text , Float  
from sqlalchemy.orm import relationship 

from .connection import Base 

class DocumentModel(Base):
    __tablename__='documents'
    
    id = Column(Integer, primary_key=True , index=True , autoincrement=True )
    doc_id = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column (DateTime, default=datetime.utcnow)

    chunks = relationship(

            "DocumentChunkModel", back_populates="document", cascade="all, delete-orphan"
    )



class DocumentChunkModel(Base):
    __tablename__="document_chunks"
    
    id = Column(Integer, primary_key=True , index=True , autoincrement=True )
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content= Column (Text, nullable=False) 
    vector_id= Column(String, unique=True , index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_hash= Column(String , unique=True , index=True , nullable=False)
    is_embedded = Column (
        Boolean , default=False , nullable=False , index=True 
    )

    document = relationship(
        "DocumentModel" , back_populates="chunks" 
        )


