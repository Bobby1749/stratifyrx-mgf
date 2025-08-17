import os, io, csv, zipfile, re, requests
from datetime import date
from dateutil import parser as dtparse

from fastapi import FastAPI, Depends, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Date, Boolean,
    UniqueConstraint, Index, select, func
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# -------------------- DB setup (Postgres preferred, SQLite fallback) --------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")
engine_kwargs = dict(pool_pre_ping=True)
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    engine_kwargs.update(pool_size=5, max_overflow=10)

engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# -------------------- Models --------------------
class Product(Base):
    __tablename__ = "ob_products"
    id = Column(Integer, primary_key=True, index=True)
    appl_type = Column(String(1), index=True)   # 'N' or 'A'
    appl_no = Column(Integer, index=True)       # NDA/ANDA
    product_no = Column(Integer, index=True)
    ingredient = Column(String)
    trade_name = Column(String, index=True)
    dosage_form = Column(String, index=True)
    route = Column(String, index=True)
    strength = Column(String)
    te_code = Column(String(8))
    approval_date = Column(Date, index=True)
    rld = Column(Boolean, default=False)
    rs = Column(Boolean, default=False)
    type = Column(String(10))                   # RX/OTC/DISCN
    applicant_short = Column(String)
    applicant_full = Column(String)
    __table_args__ = (
        UniqueConstraint("appl_no", "product_no", name="uq_products_appl_product"),
        Index("idx_products_name_ing", "trade_name", "ingredient"),
    )

class Patent(Base):
    __tablename__ = "ob_patents"
    id = Column(Integer, primary_key=True, index=True)
    appl_type = Column(String(1), index=True)
    appl_no = Column(Integer, index=True)
    product_no = Column(Integer, index=True)
    patent_no = Column(String(20), index=True)
    patent_expire_date = Column(Date, index=True)
    drug_substance_flag = Column(Boolean, default=None)
    drug_product_flag = Column(Boolean, default=None)
    patent_use_code = Column(String(20))
    delist_flag = Column(Boolean, default=None)
    patent_submission_date = Column(Date)
    __table_args__ = (
        UniqueConstraint("appl_no", "product_no", "patent_no", name="uq_patent_key"),
        Index("idx_patent_exp", "appl_no", "product_no", "patent_expire_date"),
    )

class Exclusivity(Base):
    __tablename__ = "ob_exclusivities"
    id = Column(Integer, primary_key=True, index=True)
    appl_type = Column(String(1), index=True)
    appl_no = Column(Integer, index=True)
    product_no = Column(Integer, index=True)
    exclusivity_code = Column(String(30), index=True)
    exclusivity_date = Column(Date, index=True)
    __table_args__ = (
        UniqueConstraint("appl_no", "product_no", "exclusivity_code", name="uq_excl_key"),
    )

def init_db():
    Base.metadata.create_all(bind=engine)

# -------------------- Orange Book loader (auto-discovers correct ZIP) --------------------
FDA_DATA_PAGE = "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"

def _to_date(val):
    if not val or not val.strip(): return None
    try: return dtparse.parse(val.strip(), dayfirst=False).date()
    except Exception: return None

def _to_bool_from_flag(val, on="Y"):
    if not val: return None
    return val.strip().upper() == on

def _to_bool_marked(val, mark="RLD"):
    if not val: return False
    return val.strip().upper() == mark

def _split_dosage_route(text):
    if not text: return None, None
    parts = text.split(";")
    if len(parts) >= 2: return parts[0].strip(), parts[1].strip()
    return text.strip(), None

def _is_zip_bytes(b: bytes) -> bool:
    return len(b) >= 2 and b[:2] == b"PK"

def _try_fetch_zip(url: str) -> bytes | None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content if _is_zip_bytes(r.content) else None

def fetch_zip_bytes() -> bytes:
    env_url = os.getenv("ORANGE_BOOK_ZIP_URL", "").strip()
    if env_url:
        b = _try_fetch_zip(env_url)
        if b: return b
    page = requests.get(FDA_DATA_PAGE, timeout=60); page.raise_for_status()
    hrefs = re.findall(r'href="([^"]+)"', page.text)
    candidates = []
    for h in hrefs:
        if "/media/" in h and "download" in h:
            candidates.append(h if h.startswith("http") else "https://www.fda.gov"+h)
    last_err = None
    for url in candidates:
        try:
            b = _try_fetch_zip(url)
            if b: return b
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not locate a valid Orange Book ZIP. Last error: {last_err}")

def download_and_extract():
    content = fetch_zip_bytes()
    z = zipfile.ZipFile(io.BytesIO(content))
    members = {name.lower(): name for name in z.namelist()}
    paths = {
        "products": next((v for k,v in members.items() if "product" in k), None),
        "patent":   next((v for k,v in members.items() if "patent" in k),  None),
        "excl":     next((v for k,v in members.items() if "exclus" in k),  None),
    }
    if not all(paths.values()):
        raise RuntimeError(f"ZIP missing FDA files: {paths}")
    return z, paths

def load_into_db(db: Session):
    z, paths = download_and_extract()
    db.query(Exclusivity).delete()
    db.query(Patent).delete()
    db.query(Product).delete()
    db.commit()

    with z.open(paths["products"]) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding="latin-1"), delimiter="~")
        for row in reader:
            if not row or len(row) < 14: continue
            ingredient = row[0].strip() or None
            dosage_form, route = _split_dosage_route(row[1] if len(row)>1 else "")
            trade_name = row[2].strip() or None
            applicant_short = row[3].strip() or None
            strength = row[4].strip() or None
            appl_type = (row[5] or "").strip().upper() or None
            appl_no = int(row[6]) if row[6].strip().isdigit() else None
            product_no = int(row[7]) if row[7].strip().isdigit() else None
            te_code = (row[8] or "").strip() or None
            approval_date = _to_date(row[9] if len(row)>9 else None)
            rld = _to_bool_marked(row[10] if len(row)>10 else "")
            rs  = _to_bool_marked(row[11] if len(row)>11 else "", mark="RS")
            typ = (row[12] or "").strip() or None
            applicant_full = (row[13] or "").strip() or None
            db.add(Product(
                appl_type=appl_type, appl_no=appl_no, product_no=product_no,
                ingredient=ingredient, trade_name=trade_name,
                dosage_form=dosage_form, route=route, strength=strength,
                te_code=te_code, approval_date=approval_date,
                rld=rld, rs=rs, type=typ,
                applicant_short=applicant_short, applicant_full=applicant_full
            ))
    db.commit()

    with z.open(paths["patent"]) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding="latin-1"), delimiter="~")
        for row in reader:
            if not row or len(row) < 10: continue
            appl_type = (row[0] or "").strip().upper() or None
            appl_no = int(row[1]) if row[1].strip().isdigit() else None
            product_no = int(row[2]) if row[2].strip().isdigit() else None
            patent_no = (row[3] or "").strip() or None
            patent_expire_date = _to_date(row[4] if len(row)>4 else None)
            drug_substance_flag = _to_bool_from_flag(row[5] if len(row)>5 else None)
            drug_product_flag   = _to_bool_from_flag(row[6] if len(row)>6 else None)
            patent_use_code = (row[7] or "").strip() or None
            delist_flag = _to_bool_from_flag(row[8] if len(row)>8 else None)
            patent_submission_date = _to_date(row[9] if len(row)>9 else None
            )
            db.add(Patent(
                appl_type=appl_type, appl_no=appl_no, product_no=product_no,
                patent_no=patent_no, patent_expire_date=patent_expire_date,
                drug_substance_flag=drug_substance_flag,
                drug_product_flag=drug_product_flag,
                patent_use_code=patent_use_code,
                delist_flag=delist_flag,
                patent_submission_date=patent_submission_date
            ))
    db.commit()

    with z.open(paths["excl"]) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding="latin-1"), delimiter="~")
        for row in reader:
            if not row or len(row) < 5: continue
            appl_type = (row[0] or "").strip().upper() or None
            appl_no   = int(row[1]) if row[1].strip().isdigit() else None
            product_no= int(row[2]) if row[2].strip().isdigit() else None
            exclusivity_code = (row[3] or "").strip() or None
            exclusivity_date = _to_date(row[4] if len(row)>4 else None)
            db.add(Exclusivity(
                appl_type=appl_type, appl_no=appl_no, product_no=product_no,
                exclusivity_code=exclusivity_code, exclusivity_date=exclusivity_date
            ))
    db.commit()

# -------------------- FastAPI --------------------
app = FastAPI(title="StratifyRx Market Gap Finder (MVP)")
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

class ProductOut(BaseModel):
    ingredient: Optional[str]
    trade_name: Optional[str]
    appl_type: Optional[str]
    appl_no: Optional[int]
    product_no: Optional[int]
    dosage_form: Optional[str]
    route: Optional[str]
    te_code: Optional[str]
    approval_date: Optional[str]
    latest_patent_expiry: Optional[str]
    class Config: from_attributes = True

@app.on_event("startup")
def startup(): init_db()

@app.post("/api/ob/refresh")
def refresh_ob(db: Session = Depends(get_db)):
    try:
        load_into_db(db)
        return {"status": "ok", "message": "Orange Book data loaded"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load Orange Book: {e}")

@app.get("/api/ob/products", response_model=list[ProductOut])
def list_products(
    q: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    latest_pat = (
        select(Patent.appl_no, Patent.product_no, func.max(Patent.patent_expire_date).label("latest_patent_expiry"))
        .group_by(Patent.appl_no, Patent.product_no)
        .subquery()
    )
    stmt = (
        select(
            Product.ingredient, Product.trade_name, Product.appl_type,
            Product.appl_no, Product.product_no, Product.dosage_form,
            Product.route, Product.te_code, Product.approval_date,
            latest_pat.c.latest_patent_expiry
        )
        .join(latest_pat, (latest_pat.c.appl_no == Product.appl_no) & (latest_pat.c.product_no == Product.product_no), isouter=True)
    )
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            (Product.trade_name.ilike(like)) |
            (Product.ingredient.ilike(like)) |
            (Product.applicant_full.ilike(like))
        )
    stmt = stmt.order_by(func.coalesce(latest_pat.c.latest_patent_expiry, Product.approval_date).desc()).limit(limit).offset(offset)
    rows = db.execute(stmt).all()

    out = []
    for (ingredient, trade_name, appl_type, appl_no, product_no, dosage_form, route, te_code, approval_date, latest_exp) in rows:
        out.append({
            "ingredient": ingredient,
            "trade_name": trade_name,
            "appl_type": appl_type,
            "appl_no": appl_no,
            "product_no": product_no,
            "dosage_form": dosage_form,
            "route": route,
            "te_code": te_code,
            "approval_date": approval_date.isoformat() if approval_date else None,
            "latest_patent_expiry": latest_exp.isoformat() if latest_exp else None,
        })
    return out
