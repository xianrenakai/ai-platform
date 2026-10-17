from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from app.database import Base


class CnnRaceRecord(Base):
    __tablename__ = "cnn_race_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    total_questions = Column(Integer, nullable=False)
    correct_count = Column(Integer, nullable=False)
    wrong_count = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class CnnDatasetRecord(Base):
    __tablename__ = "cnn_dataset_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    dataset_name = Column(String(200), nullable=False)
    digits = Column(Text, nullable=False)
    counts = Column(Text, nullable=False)
    total_images = Column(Integer, nullable=False)
    class_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class CnnModelDesignRecord(Base):
    __tablename__ = "cnn_model_design_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_name = Column(String(200), nullable=False)
    conv_layers = Column(Integer, nullable=False, default=0)
    conv_filters = Column(Text, nullable=False, default="[]")
    total_conv_filters = Column(Integer, nullable=False, default=0)
    pool_layers = Column(Integer, nullable=False, default=0)
    fc_neurons = Column(Integer, nullable=False, default=0)
    fc_activation = Column(String(50), nullable=False, default="relu")
    output_neurons = Column(Integer, nullable=False, default=10)
    structure_summary = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class CnnModelUploadRecord(Base):
    __tablename__ = "cnn_model_upload_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_id = Column(String(100), unique=True, nullable=False)
    model_filename = Column(String(200), nullable=False)
    model_path = Column(String(500), nullable=False)
    final_accuracy = Column(Float)
    final_val_accuracy = Column(Float)
    test_accuracy = Column(Float)
    epochs = Column(Integer)
    conv_layers = Column(Integer)
    pool_layers = Column(Integer)
    dense_layers = Column(Integer)
    total_params = Column(Integer)
    train_data_json = Column(Text)
    class_mapping_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class CnnRecognitionRecord(Base):
    __tablename__ = "cnn_recognition_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_upload_id = Column(Integer, ForeignKey("cnn_model_upload_records.id"), nullable=False)
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    image_format = Column(String(50), default="白底黑字")
    created_at = Column(DateTime, default=datetime.utcnow)
