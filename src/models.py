from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Boolean, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass

class Question(Base):
    __tablename__ = 'questions'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_embeddings: Mapped[list[float]] = mapped_column(Vector(dim=384), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    #Relationship
    answers: Mapped[list['Answer']] = relationship(back_populates='question')

class Answer(Base):
    __tablename__ = 'answers'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(ForeignKey('questions.id'), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    #Relationship
    question: Mapped['Question'] = relationship(back_populates='answers')
