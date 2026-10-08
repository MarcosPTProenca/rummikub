from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Peca(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True) 
    
    id: str
    cor: Literal["azul", "vermelho", "verde", "amarelo", "coringa"]
    numero: int = Field(ge=0, le=13)

    @model_validator(mode="after")
    def validar_coringa(self) -> "Peca":
        if (self.cor == "coringa") != (self.numero == 0):
            raise ValueError("Somente coringas têm número 0; peças normais usam 1 a 13.")
        return self

if __name__ == "__main__":

    peca_1 = Peca(id='1', cor='azul', numero=13)
