from pydantic import BaseModel, Field
from .peca import Peca

class Acao(BaseModel):
    
    tipo: str
    id_pecas: list[Peca] = Field(default_factory=list)