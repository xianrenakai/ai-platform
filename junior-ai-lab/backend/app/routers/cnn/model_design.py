import json

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.models.cnn.models import CnnModelDesignRecord

router = APIRouter(prefix="/api/cnn/model-design", tags=["cnn-model-design"])


class ModelDesignSaveRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=200)
    conv_layers: int = Field(..., ge=0)
    conv_filters: list[int] = Field(default_factory=list)
    total_conv_filters: int = Field(..., ge=0)
    pool_layers: int = Field(..., ge=0)
    fc_neurons: int = Field(..., ge=0)
    fc_activation: str = Field(default="relu", max_length=50)
    output_neurons: int = Field(..., ge=1)
    structure_summary: str = ""


@router.post("/save", status_code=status.HTTP_201_CREATED)
def save_model_design(
    body: ModelDesignSaveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db.add(CnnModelDesignRecord(
        user_id=current_user.id,
        model_name=body.model_name,
        conv_layers=body.conv_layers,
        conv_filters=json.dumps(body.conv_filters),
        total_conv_filters=body.total_conv_filters,
        pool_layers=body.pool_layers,
        fc_neurons=body.fc_neurons,
        fc_activation=body.fc_activation,
        output_neurons=body.output_neurons,
        structure_summary=body.structure_summary,
    ))
    db.commit()
    return {"ok": True, "message": "模型设计保存成功"}
