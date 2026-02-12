# Rasyn API — Complete Developer Integration Guide

> **For**: Frontend developer integrating rasyn.ai with the Rasyn Retrosynthesis API
> **Version**: 2.0.0
> **Last updated**: 2026-02-12
> **Author**: Ansh (backend/ML) — for the frontend team

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Setting Up api.rasyn.ai (DNS + HTTPS)](#2-setting-up-apirasynaì-dns--https)
3. [Authentication System](#3-authentication-system)
4. [API Key Management (User Self-Service)](#4-api-key-management-user-self-service)
5. [CORS Configuration](#5-cors-configuration)
6. [Rate Limits](#6-rate-limits)
7. [API Endpoints — Complete Reference](#7-api-endpoints--complete-reference)
8. [Error Handling — Every Error Code Explained](#8-error-handling--every-error-code-explained)
9. [Frontend Integration — Full Next.js Implementation](#9-frontend-integration--full-nextjs-implementation)
10. [Gradio Demo — Authentication & Access](#10-gradio-demo--authentication--access)
11. [User Onboarding & Subscription Flow](#11-user-onboarding--subscription-flow)
12. [Performance, Timeouts & UX](#12-performance-timeouts--ux)
13. [Security Checklist — Before Going Live](#13-security-checklist--before-going-live)
14. [AWS Infrastructure Reference](#14-aws-infrastructure-reference)
15. [cURL Examples — Quick Testing](#15-curl-examples--quick-testing)
16. [FAQ & Troubleshooting](#16-faq--troubleshooting)

---

## 1. Overview & Architecture

Rasyn provides AI-powered retrosynthetic analysis through a REST API. Given a target molecule (as a SMILES string), the API predicts how to synthesize it — either one step back (single-step) or a full route to purchasable starting materials (multi-step).

### What is SMILES?

SMILES (Simplified Molecular-Input Line-Entry System) is a text representation of chemical structures. For example:
- `CC(=O)Oc1ccccc1C(=O)O` = Aspirin
- `CC(=O)Nc1ccc(O)cc1` = Acetaminophen (Tylenol)
- `c1ccccc1` = Benzene

Users will type these into your frontend (or use a molecule editor that outputs SMILES).

### AI Models Available

| Model | ID | Speed | Quality | Description |
|-------|-----|-------|---------|-------------|
| **RetroTransformer v2** | `retro` | Fast (~0.2–1s) | Good | Specialized seq2seq model with copy mechanism |
| **LLM (RSGPT v6)** | `llm` | Slow (~3–200s) | Better | Large language model fine-tuned on reactions, includes round-trip verification |
| **Both** | `both` | Slow | Best | Runs both models, merges and re-ranks results |

### System Architecture

```
┌──────────────────────┐         ┌───────────────────────┐
│  Your Frontend App   │         │   Rasyn API Backend    │
│  (rasyn.ai)          │         │   (api.rasyn.ai)       │
│                      │         │                        │
│  Next.js / React     │         │  EC2 g5.xlarge (A10G)  │
│  ┌─────────────┐     │  HTTPS  │  ┌──────────────────┐  │
│  │ Browser SPA │────────────────→ │  ALB (port 443)  │  │
│  └─────────────┘     │         │  └────────┬─────────┘  │
│         │            │         │           │             │
│  ┌──────▼──────┐     │         │  ┌────────▼─────────┐  │
│  │ Next.js API │────────────────→ │  FastAPI (8000)   │  │
│  │  Routes     │     │         │  │  ├── Auth MW      │  │
│  │ (adds key)  │     │         │  │  ├── Rate Limit   │  │
│  └─────────────┘     │         │  │  └── Routes       │  │
│                      │         │  │      ├── /retro/*  │  │
│  .env.local:         │         │  │      ├── /mols/*   │  │
│  RASYN_API_KEY=...   │         │  │      └── /keys/*   │  │
│  RASYN_ADMIN_KEY=... │         │  ├──────────────────┤  │
│                      │         │  │  Service Layer     │  │
│                      │         │  │  ├── Pipeline      │  │
│                      │         │  │  └── ModelManager  │  │
│                      │         │  ├──────────────────┤  │
│                      │         │  │  GPU Models        │  │
│                      │         │  │  ├── LLM (RSGPT)  │  │
│                      │         │  │  ├── RetroTx v2   │  │
│                      │         │  │  ├── Forward       │  │
│                      │         │  │  └── Graph Head    │  │
│                      │         │  └──────────────────┘  │
└──────────────────────┘         └───────────────────────┘
```

**Critical rule**: The Rasyn API key (`rsy_...`) NEVER goes to the browser. It lives ONLY in your Next.js backend (server-side API routes or middleware). The browser talks to YOUR server, YOUR server talks to the Rasyn API with the key.

---

## 2. Setting Up api.rasyn.ai (DNS + HTTPS)

The API backend runs on AWS behind an Application Load Balancer (ALB). To make it accessible at `api.rasyn.ai`, you need to:

### Step 1: Validate the SSL Certificate (ACM)

An SSL certificate has already been requested via AWS Certificate Manager for `api.rasyn.ai`. To validate it, you need to add ONE DNS record.

**Go to your DNS provider** (wherever rasyn.ai is managed — Cloudflare, Route53, Namecheap, etc.) and add:

| Record Type | Name (Host) | Value (Points To) | TTL |
|-------------|-------------|---------------------|-----|
| **CNAME** | `_21f7ed329fffff89e760e0a1716a6f1e.api.rasyn.ai` | `_a14be9ea59fed95b98c5ab37ac49aed6.jkddzztszm.acm-validations.aws.` | 300 |

> **Cloudflare users**: Make sure the proxy is OFF (DNS Only / grey cloud) for this validation record.

> **Namecheap users**: Remove `.api.rasyn.ai` from the Name field — Namecheap appends the domain automatically. So the host would be `_21f7ed329fffff89e760e0a1716a6f1e.api`.

After adding this record, AWS will verify it within 5–30 minutes. You can check the status:
- AWS Console → Certificate Manager → Look for `api.rasyn.ai` → Status should change from `Pending validation` to `Issued`
- Or ask Ansh to run: `aws acm describe-certificate --certificate-arn arn:aws:acm:us-east-1:779530687716:certificate/9e0f3fef-6e47-4896-bc9b-72d01c3eef5e --query 'Certificate.Status'`

### Step 2: Point api.rasyn.ai to the Load Balancer

Add another DNS record:

| Record Type | Name (Host) | Value (Points To) | TTL |
|-------------|-------------|---------------------|-----|
| **CNAME** | `api.rasyn.ai` | `rasyn-prod-alb-449951296.us-east-1.elb.amazonaws.com` | 300 |

> **Cloudflare users**: You can enable the proxy (orange cloud) for this record for extra DDoS protection, but make sure SSL mode is set to "Full (strict)" in Cloudflare settings.

> **Namecheap users**: Host = `api`, Target = the ALB URL above.

After DNS propagation (usually 1–10 minutes), `api.rasyn.ai` will resolve to the ALB.

### Step 3: Create HTTPS Listener on ALB

Once the certificate status is `ISSUED`, ask Ansh to run these commands (or run them yourself if you have AWS CLI access):

```bash
# Create HTTPS listener (port 443)
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:779530687716:loadbalancer/app/rasyn-prod-alb/53d5b0dd8df7ab1f \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:us-east-1:779530687716:certificate/9e0f3fef-6e47-4896-bc9b-72d01c3eef5e \
  --default-actions "Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:779530687716:targetgroup/rasyn-prod-tg/20b35741cfb1023c" \
  --region us-east-1
```

### Step 4: Redirect HTTP → HTTPS

```bash
# Find the HTTP listener ARN
HTTP_ARN=$(aws elbv2 describe-listeners \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:779530687716:loadbalancer/app/rasyn-prod-alb/53d5b0dd8df7ab1f \
  --query "Listeners[?Port==\`80\`].ListenerArn" --output text --region us-east-1)

# Redirect all HTTP to HTTPS
aws elbv2 modify-listener \
  --listener-arn $HTTP_ARN \
  --default-actions "Type=redirect,RedirectConfig={Protocol=HTTPS,Port=443,StatusCode=HTTP_301}" \
  --region us-east-1
```

### Step 5: Verify

```bash
# Should return {"status": "ok", ...}
curl https://api.rasyn.ai/api/v1/health

# Should redirect to HTTPS
curl -I http://api.rasyn.ai/api/v1/health
# → 301 Location: https://api.rasyn.ai/api/v1/health
```

### DNS Summary — All Records Needed

| Type | Name | Value | Purpose |
|------|------|-------|---------|
| CNAME | `_21f7ed329fffff89e760e0a1716a6f1e.api` | `_a14be9ea59fed95b98c5ab37ac49aed6.jkddzztszm.acm-validations.aws.` | SSL cert validation |
| CNAME | `api` | `rasyn-prod-alb-449951296.us-east-1.elb.amazonaws.com` | API subdomain |

---

## 3. Authentication System

### How It Works

The API uses **API key authentication**. Every request to `/api/v1/*` endpoints (except `/api/v1/health`) must include a valid API key.

Keys are stored as SHA-256 hashes in a SQLite database on the server. Two roles exist:
- **admin**: Can call all API endpoints AND manage keys (create/list/revoke)
- **user**: Can only call API endpoints (retro, molecules, etc.)

### Three Ways to Pass an API Key

#### Method 1: `X-API-Key` Header (Recommended)

```http
POST /api/v1/retro/single-step HTTP/1.1
Host: api.rasyn.ai
Content-Type: application/json
X-API-Key: rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq

{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
```

#### Method 2: `Authorization: Bearer` Header

```http
POST /api/v1/retro/single-step HTTP/1.1
Host: api.rasyn.ai
Content-Type: application/json
Authorization: Bearer rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq

{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
```

#### Method 3: `api_key` Query Parameter

```
GET /api/v1/molecules/image?smiles=c1ccccc1&api_key=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq
```

### Current Keys

| Key | Role | Purpose |
|-----|------|---------|
| `rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq` | **admin** | Master key — use on your backend to create user keys |

> **CRITICAL**: This admin key must NEVER appear in client-side JavaScript, HTML, or any code that reaches the browser. Store it in your server's environment variables (`.env.local` for Next.js).

### Your `.env.local` File

```env
# Master admin key — used by YOUR backend to manage user keys
RASYN_ADMIN_KEY=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq

# Base URL (update once DNS is set up)
RASYN_API_URL=https://api.rasyn.ai

# If using a single shared key for all API calls (simpler but less trackable):
RASYN_API_KEY=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq
```

### Auth Error Responses

**401 — No key provided:**
```json
{
  "error": "authentication_required",
  "message": "API key required. Pass via X-API-Key header, Authorization: Bearer <key>, or ?api_key= query param."
}
```

**403 — Invalid or revoked key:**
```json
{
  "error": "invalid_api_key",
  "message": "The provided API key is not valid or has been revoked."
}
```

**403 — Non-admin key on admin endpoint:**
```json
{
  "error": "admin_required",
  "message": "This endpoint requires an admin API key."
}
```

---

## 4. API Key Management (User Self-Service)

These endpoints let your backend create per-user API keys, track usage, and revoke keys. **All require an admin API key.**

### 4.1 Create a Key

```
POST /api/v1/keys
```

**Headers:** `X-API-Key: <admin_key>`

**Request Body:**
```json
{
  "name": "user-john@example.com",
  "role": "user"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | **required** | Label for this key (e.g., user email, team name). Max 100 chars. |
| `role` | string | `"user"` | `"user"` (API access only) or `"admin"` (can manage other keys too) |

**Response (201):**
```json
{
  "id": "0301bb33a1eaef71",
  "key": "rsy_EShPbqoAzq_4sTjSKy6TlEuxYBTumrwmFHU8YqkhcKdM6teB",
  "name": "user-john@example.com",
  "role": "user",
  "created_at": "2026-02-12T21:45:29.902327+00:00"
}
```

> **IMPORTANT**: The `key` field is ONLY returned once, at creation time. We store only the SHA-256 hash. If the user loses their key, you must create a new one and revoke the old one.

**TypeScript implementation for your backend:**

```typescript
// lib/rasyn-admin.ts — Server-side only!

const RASYN_API_URL = process.env.RASYN_API_URL!;
const RASYN_ADMIN_KEY = process.env.RASYN_ADMIN_KEY!;

export async function createRasynKey(userEmail: string): Promise<{
  id: string;
  key: string;
  name: string;
  role: string;
  created_at: string;
}> {
  const res = await fetch(`${RASYN_API_URL}/api/v1/keys`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": RASYN_ADMIN_KEY,
    },
    body: JSON.stringify({
      name: `user-${userEmail}`,
      role: "user",
    }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(`Failed to create Rasyn key: ${res.status} ${JSON.stringify(error)}`);
  }

  return res.json();
}
```

### 4.2 List All Keys

```
GET /api/v1/keys
```

**Headers:** `X-API-Key: <admin_key>`

**Response:**
```json
{
  "keys": [
    {
      "id": "0301bb33a1eaef71",
      "name": "user-john@example.com",
      "role": "user",
      "created_at": "2026-02-12T21:45:29.902327+00:00",
      "last_used_at": "2026-02-12T22:01:15.123456+00:00",
      "request_count": 47,
      "is_active": 1,
      "created_by": "Admin (env)"
    },
    {
      "id": "a81b2c3d4e5f6789",
      "name": "user-jane@example.com",
      "role": "user",
      "created_at": "2026-02-12T20:00:00.000000+00:00",
      "last_used_at": null,
      "request_count": 0,
      "is_active": 1,
      "created_by": "Admin (env)"
    }
  ],
  "total": 2
}
```

**Useful fields for your admin dashboard:**
- `request_count` — total API calls made with this key (for billing)
- `last_used_at` — when the key was last used (null = never)
- `is_active` — 1 = active, 0 = revoked

**TypeScript:**

```typescript
export async function listRasynKeys(): Promise<{
  keys: Array<{
    id: string;
    name: string;
    role: string;
    created_at: string;
    last_used_at: string | null;
    request_count: number;
    is_active: number;
    created_by: string | null;
  }>;
  total: number;
}> {
  const res = await fetch(`${RASYN_API_URL}/api/v1/keys`, {
    headers: { "X-API-Key": RASYN_ADMIN_KEY },
  });

  if (!res.ok) throw new Error(`Failed to list keys: ${res.status}`);
  return res.json();
}
```

### 4.3 Revoke a Key

```
DELETE /api/v1/keys/{key_id}
```

**Headers:** `X-API-Key: <admin_key>`

**Response:**
```json
{
  "revoked": true,
  "key_id": "0301bb33a1eaef71"
}
```

Revoked keys stop working within 30 seconds (cache refresh interval on the server).

**TypeScript:**

```typescript
export async function revokeRasynKey(keyId: string): Promise<boolean> {
  const res = await fetch(`${RASYN_API_URL}/api/v1/keys/${keyId}`, {
    method: "DELETE",
    headers: { "X-API-Key": RASYN_ADMIN_KEY },
  });

  if (!res.ok) throw new Error(`Failed to revoke key: ${res.status}`);
  const data = await res.json();
  return data.revoked;
}
```

### Key Roles Reference

| Role | Call retro/molecules API | Create keys | List keys | Revoke keys |
|------|--------------------------|-------------|-----------|-------------|
| `user` | Yes | No | No | No |
| `admin` | Yes | Yes | Yes | Yes |

### Two Approaches for Frontend API Keys

**Approach A: Single shared key (simpler)**
- Store one API key (`RASYN_API_KEY`) in your backend's `.env`
- All users share this key; you track usage in your own DB
- Pros: Simpler, fewer keys to manage
- Cons: Can't revoke per-user, can't track per-user usage on Rasyn side

**Approach B: Per-user keys (recommended for billing)**
- When a user signs up, call `POST /api/v1/keys` to create a key for them
- Store the user's key in YOUR database (encrypted)
- When the user makes an API call, your backend uses their specific key
- Pros: Per-user usage tracking, per-user revocation, better billing data
- Cons: More code, must handle key storage securely

---

## 5. CORS Configuration

The API server allows requests from these origins:

```
https://rasyn.ai
https://www.rasyn.ai
https://app.rasyn.ai
```

If your frontend runs on a different domain (e.g., `localhost:3000` during development), you'll get CORS errors when calling the API directly from the browser. **This is why you should use a backend proxy** (Next.js API routes) — server-to-server calls don't have CORS restrictions.

**Allowed methods:** `GET`, `POST`, `OPTIONS`
**Allowed headers:** `Content-Type`, `Authorization`, `X-API-Key`
**Exposed headers:** `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

To add more allowed origins (e.g., staging), Ansh can set the `RASYN_CORS_ORIGINS` environment variable on the EC2 instance:

```bash
ssh -i ~/.ssh/rasyn-key.pem ubuntu@44.192.125.177
sudo systemctl edit rasyn
# Add under [Service]:
# Environment=RASYN_CORS_ORIGINS=https://rasyn.ai,https://www.rasyn.ai,https://app.rasyn.ai,https://staging.rasyn.ai,http://localhost:3000
sudo systemctl restart rasyn
```

---

## 6. Rate Limits

Per-IP rate limiting protects the GPU from overload.

| Endpoint Pattern | Limit | Window | Notes |
|------------------|-------|--------|-------|
| `POST /api/v1/retro/single-step` | 20 | 60s | GPU-intensive |
| `POST /api/v1/retro/multi-step` | 5 | 60s | Very GPU-intensive, takes 10-60s per call |
| `/api/v1/molecules/*` | 60 | 60s | Lightweight (RDKit only) |
| All other `/api/*` | 60 | 60s | Default |
| `/demo/*` | Exempt | — | Gradio makes many internal calls per page load |
| `/api/v1/health` | Unlimited | — | Health check |

### Rate Limit Headers

Every response includes these headers:

```http
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 17
X-RateLimit-Reset: 1739395200
```

### 429 Too Many Requests

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Max 20 requests per 60s for this endpoint.",
  "retry_after": "45"
}
```

### Frontend Handling

```typescript
// In your Next.js API route (proxy to Rasyn)
const res = await fetch(`${RASYN_API_URL}/api/v1/retro/single-step`, { ... });

if (res.status === 429) {
  const retryAfter = res.headers.get("Retry-After") || "60";
  return NextResponse.json(
    {
      error: "rate_limited",
      message: `Too many requests. Please wait ${retryAfter} seconds.`,
      retryAfter: parseInt(retryAfter),
    },
    { status: 429 }
  );
}
```

On the frontend, show a user-friendly message and optionally implement a countdown timer.

---

## 7. API Endpoints — Complete Reference

### Base URL

| Environment | URL |
|-------------|-----|
| Production | `https://api.rasyn.ai` (after DNS setup) |
| Direct ALB (temporary) | `http://rasyn-prod-alb-449951296.us-east-1.elb.amazonaws.com` |
| Local dev | `http://localhost:8000` |

### 7.1 Health Check

```
GET /api/v1/health
```

**No authentication required.** Use this to check if the API is up.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": ["retro_v2", "llm", "forward"],
  "device": "cuda"
}
```

`models_loaded` only shows models that have been lazy-loaded (called at least once). On a fresh restart, this will be `[]`. Don't worry — models load automatically on first API call.

---

### 7.2 Single-Step Retrosynthesis

Given a product molecule, predict reactants that could produce it (one step backwards).

```
POST /api/v1/retro/single-step
```

**Request:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "model": "retro",
  "top_k": 10,
  "use_verification": true
}
```

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `smiles` | string | — | **Yes** | Product molecule in SMILES notation |
| `model` | string | `"llm"` | No | `"llm"`, `"retro"`, or `"both"` |
| `top_k` | integer | `10` | No | Number of predictions to return (1–50) |
| `use_verification` | boolean | `true` | No | Run forward-model verification (LLM only) |

**Response:**
```json
{
  "product": "CC(=O)Oc1ccccc1C(=O)O",
  "predictions": [
    {
      "rank": 1,
      "reactants_smiles": ["O=C(O)c1ccccc1O", "CC(=O)Cl"],
      "confidence": 0.8043,
      "model_source": "retro_v2",
      "verification": {
        "rdkit_valid": true,
        "forward_match_score": 0.95,
        "overall_confidence": 0.8043
      },
      "edit_info": {
        "bonds": [[2, 7]],
        "synthons": ["O=C(O)c1ccccc1[OH]", "[CH3]C(=O)[Cl]"],
        "leaving_groups": ["[OH]", "[Cl]"]
      }
    },
    {
      "rank": 2,
      "reactants_smiles": ["O=C(O)c1ccccc1O", "CC(=O)OC(C)=O"],
      "confidence": 0.6234,
      "model_source": "retro_v2",
      "verification": null,
      "edit_info": null
    }
  ],
  "compute_time_ms": 245.3,
  "error": null
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `product` | string | Canonical SMILES of the input |
| `predictions` | array | Ranked list of retrosynthetic predictions |
| `predictions[].rank` | int | 1-indexed rank |
| `predictions[].reactants_smiles` | string[] | Reactant SMILES (what you need to buy/synthesize) |
| `predictions[].confidence` | float | Model confidence 0.0–1.0 |
| `predictions[].model_source` | string | `"llm"` or `"retro_v2"` |
| `predictions[].verification` | object\|null | Forward-model check results (LLM only) |
| `predictions[].verification.rdkit_valid` | bool | Are the reactants chemically valid? |
| `predictions[].verification.forward_match_score` | float | How well do reactants reproduce the product? (0–1) |
| `predictions[].edit_info` | object\|null | Bond disconnection details (when available) |
| `compute_time_ms` | float | Server compute time in milliseconds |
| `error` | string\|null | Error message if something failed |

**Model selection guide:**

| Use Case | Model | Why |
|----------|-------|-----|
| Interactive UI, fast feedback | `"retro"` | Returns in <1s, good for live typing |
| High-quality analysis | `"llm"` | Better predictions with verification |
| Full comparison for research | `"both"` | Shows results from both models |

---

### 7.3 Multi-Step Route Planning

Plan a full synthetic route from target molecule back to purchasable starting materials.

```
POST /api/v1/retro/multi-step
```

**Request:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "max_depth": 10,
  "max_routes": 5
}
```

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `smiles` | string | — | **Yes** | Target molecule SMILES |
| `max_depth` | integer | `10` | No | Max retrosynthetic steps deep (1–20) |
| `max_routes` | integer | `5` | No | Max number of complete routes (1–20) |

**Response:**
```json
{
  "target": "CC(=O)Oc1ccccc1C(=O)O",
  "routes": [
    {
      "steps": [
        {
          "product": "CC(=O)Oc1ccccc1C(=O)O",
          "reactants": ["O=C(O)c1ccccc1O", "CC(=O)Cl"],
          "confidence": 0.85
        },
        {
          "product": "O=C(O)c1ccccc1O",
          "reactants": ["O=Cc1ccccc1O"],
          "confidence": 0.72
        }
      ],
      "total_score": 1.57,
      "num_steps": 2,
      "all_available": true,
      "starting_materials": ["CC(=O)Cl", "O=Cc1ccccc1O"]
    }
  ],
  "compute_time_ms": 49377.0,
  "error": null
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `target` | string | Canonical SMILES of the target |
| `routes` | array | List of complete routes, best first |
| `routes[].steps` | array | Steps from target → starting materials |
| `routes[].steps[].product` | string | What this step produces |
| `routes[].steps[].reactants` | string[] | What you need for this step |
| `routes[].steps[].confidence` | float | Model confidence (0–1) |
| `routes[].total_score` | float | Aggregate route quality (higher = better) |
| `routes[].num_steps` | int | Number of synthetic steps |
| `routes[].all_available` | bool | Are ALL starting materials purchasable? |
| `routes[].starting_materials` | string[] | Final leaf-node molecules |
| `compute_time_ms` | float | Server compute time (ms) |

> **Warning:** Multi-step is computationally expensive. Typical response times are **10–120 seconds**. Set client timeouts to at least 10 minutes (see Section 12).

---

### 7.4 Validate SMILES

Validate a SMILES string and get molecular information + structure image.

```
POST /api/v1/molecules/validate
```

**Request:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(O)=O"
}
```

**Response:**
```json
{
  "valid": true,
  "canonical": "CC(=O)Oc1ccccc1C(=O)O",
  "formula": "C9H8O4",
  "mol_weight": 180.042,
  "svg": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `valid` | bool | Is the SMILES chemically valid? |
| `canonical` | string\|null | Canonical (standardized) SMILES |
| `formula` | string\|null | Molecular formula (e.g., C9H8O4) |
| `mol_weight` | float\|null | Molecular weight in g/mol |
| `svg` | string\|null | Base64-encoded SVG of the molecule structure |

**Use this for:**
- Input validation before calling retrosynthesis
- Showing molecule previews as the user types
- Displaying molecular info (formula, weight) in your UI

**Rendering the SVG in React:**
```tsx
function MoleculeImage({ svg }: { svg: string | null }) {
  if (!svg) return null;
  return (
    <img
      src={`data:image/svg+xml;base64,${svg}`}
      alt="Molecule structure"
      className="w-48 h-48"
    />
  );
}
```

---

### 7.5 Molecule Image (Direct SVG)

Get a rendered SVG image of a molecule directly (useful for `<img>` tags).

```
GET /api/v1/molecules/image?smiles=CC(=O)Oc1ccccc1C(=O)O
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `smiles` | string (query) | Yes | URL-encoded SMILES string |

**Response:** SVG image (`Content-Type: image/svg+xml`)

**Error:** HTTP 400 with text `"Invalid SMILES"` if the SMILES is invalid.

---

## 8. Error Handling — Every Error Code Explained

### HTTP Status Codes

| Code | Name | When It Happens | What to Do |
|------|------|-----------------|------------|
| **200** | OK | Request succeeded | Display results |
| **400** | Bad Request | Invalid SMILES, malformed JSON | Show "Invalid molecule" to user |
| **401** | Unauthorized | Missing API key | Check your backend `.env` key |
| **403** | Forbidden | Invalid/revoked key, or non-admin on admin endpoint | Check key validity, generate a new one |
| **422** | Validation Error | Request body doesn't match schema (e.g., missing `smiles`) | Fix request format |
| **429** | Too Many Requests | Rate limit exceeded | Wait and retry (see `retry_after`) |
| **500** | Internal Server Error | Model crashed, GPU OOM, unexpected bug | Show "Service temporarily unavailable", log the error |
| **502** | Bad Gateway | ALB can't reach the backend | Backend is down or restarting — wait a minute |
| **504** | Gateway Timeout | ALB timed out waiting for backend | Request took too long — try `"retro"` model (faster) |

### 422 Validation Error Format

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "smiles"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

### Complete Error Handling (TypeScript)

```typescript
// lib/rasyn-client.ts — Server-side Rasyn API client

const RASYN_API_URL = process.env.RASYN_API_URL!;

export class RasynAPIError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public retryAfter?: number,
  ) {
    super(message);
    this.name = "RasynAPIError";
  }
}

export async function callRasynAPI<T>(
  endpoint: string,
  options: {
    method?: "GET" | "POST" | "DELETE";
    body?: object;
    apiKey: string;
    timeoutMs?: number;
  }
): Promise<T> {
  const { method = "POST", body, apiKey, timeoutMs = 300_000 } = options;

  const headers: Record<string, string> = {
    "X-API-Key": apiKey,
  };
  if (body) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`${RASYN_API_URL}${endpoint}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(timeoutMs),
  });

  // Handle specific error codes
  if (res.status === 401) {
    throw new RasynAPIError(401, "auth_required", "API key is missing");
  }
  if (res.status === 403) {
    const data = await res.json().catch(() => ({}));
    throw new RasynAPIError(403, data.error || "forbidden", data.message || "Access denied");
  }
  if (res.status === 429) {
    const data = await res.json().catch(() => ({}));
    const retryAfter = parseInt(res.headers.get("Retry-After") || "60");
    throw new RasynAPIError(429, "rate_limited", data.message || "Rate limited", retryAfter);
  }
  if (res.status === 422) {
    const detail = await res.json().catch(() => ({}));
    throw new RasynAPIError(422, "validation_error", JSON.stringify(detail));
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new RasynAPIError(res.status, "api_error", text);
  }

  return res.json();
}
```

---

## 9. Frontend Integration — Full Next.js Implementation

### File Structure

```
your-nextjs-app/
├── .env.local                          # API keys (never committed to git!)
├── lib/
│   ├── rasyn-client.ts                 # Server-side Rasyn API client
│   └── rasyn-admin.ts                  # Admin key management
├── app/
│   ├── api/
│   │   ├── retro/
│   │   │   ├── single-step/route.ts    # Proxy: single-step retro
│   │   │   └── multi-step/route.ts     # Proxy: multi-step retro
│   │   ├── molecules/
│   │   │   ├── validate/route.ts       # Proxy: SMILES validation
│   │   │   └── image/route.ts          # Proxy: molecule image
│   │   └── keys/
│   │       └── route.ts                # User key management
│   └── dashboard/
│       └── page.tsx                    # Main retrosynthesis UI
├── hooks/
│   ├── useRetrosynthesis.ts            # React hook for retro calls
│   └── useMoleculeValidation.ts        # React hook for validation
└── components/
    ├── RetroSearch.tsx                 # Main search component
    ├── MoleculeImage.tsx               # SVG molecule renderer
    ├── ResultsTable.tsx                # Predictions table
    └── RouteViewer.tsx                 # Multi-step route display
```

### `.env.local`

```env
# Rasyn API Configuration
RASYN_API_URL=https://api.rasyn.ai
RASYN_ADMIN_KEY=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq
RASYN_API_KEY=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq
```

> Add `.env.local` to your `.gitignore`!

### `lib/rasyn-client.ts` — Server-Side API Client

```typescript
// lib/rasyn-client.ts
// This file runs ONLY on the server (Next.js API routes).
// It holds the API key and makes authenticated calls to Rasyn.

const RASYN_API_URL = process.env.RASYN_API_URL || "https://api.rasyn.ai";
const RASYN_API_KEY = process.env.RASYN_API_KEY!;

export class RasynError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public retryAfter?: number,
  ) {
    super(message);
  }
}

async function rasynFetch<T>(
  path: string,
  opts: { method?: string; body?: object; timeoutMs?: number; apiKey?: string } = {}
): Promise<T> {
  const { method = "POST", body, timeoutMs = 300_000, apiKey } = opts;
  const key = apiKey || RASYN_API_KEY;

  const headers: Record<string, string> = { "X-API-Key": key };
  if (body) headers["Content-Type"] = "application/json";

  const res = await fetch(`${RASYN_API_URL}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(timeoutMs),
  });

  if (res.status === 401) throw new RasynError(401, "auth", "API key missing");
  if (res.status === 403) throw new RasynError(403, "forbidden", "Invalid API key");
  if (res.status === 429) {
    const ra = parseInt(res.headers.get("Retry-After") || "60");
    throw new RasynError(429, "rate_limited", `Rate limited, retry in ${ra}s`, ra);
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new RasynError(res.status, "error", text || `HTTP ${res.status}`);
  }

  return res.json();
}

// --- Public API ---

export interface SingleStepResult {
  product: string;
  predictions: Array<{
    rank: number;
    reactants_smiles: string[];
    confidence: number;
    model_source: string;
    verification: {
      rdkit_valid: boolean;
      forward_match_score: number;
      overall_confidence: number;
    } | null;
    edit_info: {
      bonds: number[][];
      synthons: string[];
      leaving_groups: string[];
    } | null;
  }>;
  compute_time_ms: number;
  error: string | null;
}

export interface MultiStepResult {
  target: string;
  routes: Array<{
    steps: Array<{
      product: string;
      reactants: string[];
      confidence: number;
    }>;
    total_score: number;
    num_steps: number;
    all_available: boolean;
    starting_materials: string[];
  }>;
  compute_time_ms: number;
  error: string | null;
}

export interface ValidateResult {
  valid: boolean;
  canonical: string | null;
  formula: string | null;
  mol_weight: number | null;
  svg: string | null;
}

export function singleStep(
  smiles: string,
  model: string = "retro",
  topK: number = 10,
  useVerification: boolean = true,
  apiKey?: string,
) {
  return rasynFetch<SingleStepResult>("/api/v1/retro/single-step", {
    body: { smiles, model, top_k: topK, use_verification: useVerification },
    timeoutMs: model === "retro" ? 30_000 : 300_000,
    apiKey,
  });
}

export function multiStep(
  smiles: string,
  maxDepth: number = 10,
  maxRoutes: number = 5,
  apiKey?: string,
) {
  return rasynFetch<MultiStepResult>("/api/v1/retro/multi-step", {
    body: { smiles, max_depth: maxDepth, max_routes: maxRoutes },
    timeoutMs: 600_000,
    apiKey,
  });
}

export function validateSmiles(smiles: string, apiKey?: string) {
  return rasynFetch<ValidateResult>("/api/v1/molecules/validate", {
    body: { smiles },
    timeoutMs: 10_000,
    apiKey,
  });
}

export async function getMoleculeImage(smiles: string, apiKey?: string): Promise<ArrayBuffer> {
  const key = apiKey || RASYN_API_KEY;
  const res = await fetch(
    `${RASYN_API_URL}/api/v1/molecules/image?smiles=${encodeURIComponent(smiles)}`,
    { headers: { "X-API-Key": key }, signal: AbortSignal.timeout(10_000) }
  );
  if (!res.ok) throw new RasynError(res.status, "image_error", "Failed to get molecule image");
  return res.arrayBuffer();
}

export function healthCheck() {
  return rasynFetch<{ status: string; models_loaded: string[]; device: string }>(
    "/api/v1/health",
    { method: "GET", timeoutMs: 5_000 }
  );
}
```

### `app/api/retro/single-step/route.ts`

```typescript
// app/api/retro/single-step/route.ts
import { NextRequest, NextResponse } from "next/server";
import { singleStep, RasynError } from "@/lib/rasyn-client";

export async function POST(request: NextRequest) {
  // TODO: Add your own user authentication here
  // const session = await getServerSession(authOptions);
  // if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  try {
    const body = await request.json();
    const { smiles, model = "retro", topK = 10 } = body;

    if (!smiles || typeof smiles !== "string") {
      return NextResponse.json({ error: "SMILES is required" }, { status: 400 });
    }

    // TODO: Check user's subscription/credits here
    // const credits = await getUserCredits(session.user.id);
    // if (credits <= 0) return NextResponse.json({ error: "No credits" }, { status: 402 });

    const result = await singleStep(smiles, model, topK);

    // TODO: Deduct credits after successful call
    // await deductCredits(session.user.id, 1);

    return NextResponse.json(result);
  } catch (err) {
    if (err instanceof RasynError) {
      return NextResponse.json(
        { error: err.code, message: err.message, retryAfter: err.retryAfter },
        { status: err.status }
      );
    }
    if (err instanceof Error && err.name === "TimeoutError") {
      return NextResponse.json({ error: "timeout", message: "Request timed out" }, { status: 504 });
    }
    console.error("Single-step error:", err);
    return NextResponse.json({ error: "internal", message: "Service unavailable" }, { status: 503 });
  }
}
```

### `app/api/retro/multi-step/route.ts`

```typescript
// app/api/retro/multi-step/route.ts
import { NextRequest, NextResponse } from "next/server";
import { multiStep, RasynError } from "@/lib/rasyn-client";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { smiles, maxDepth = 10, maxRoutes = 5 } = body;

    if (!smiles || typeof smiles !== "string") {
      return NextResponse.json({ error: "SMILES is required" }, { status: 400 });
    }

    const result = await multiStep(smiles, maxDepth, maxRoutes);
    return NextResponse.json(result);
  } catch (err) {
    if (err instanceof RasynError) {
      return NextResponse.json(
        { error: err.code, message: err.message, retryAfter: err.retryAfter },
        { status: err.status }
      );
    }
    if (err instanceof Error && err.name === "TimeoutError") {
      return NextResponse.json({ error: "timeout", message: "Request timed out" }, { status: 504 });
    }
    console.error("Multi-step error:", err);
    return NextResponse.json({ error: "internal", message: "Service unavailable" }, { status: 503 });
  }
}
```

### `app/api/molecules/validate/route.ts`

```typescript
// app/api/molecules/validate/route.ts
import { NextRequest, NextResponse } from "next/server";
import { validateSmiles, RasynError } from "@/lib/rasyn-client";

export async function POST(request: NextRequest) {
  try {
    const { smiles } = await request.json();
    if (!smiles) return NextResponse.json({ error: "SMILES required" }, { status: 400 });

    const result = await validateSmiles(smiles);
    return NextResponse.json(result);
  } catch (err) {
    if (err instanceof RasynError) {
      return NextResponse.json({ error: err.message }, { status: err.status });
    }
    return NextResponse.json({ error: "Validation failed" }, { status: 500 });
  }
}
```

### `app/api/molecules/image/route.ts`

```typescript
// app/api/molecules/image/route.ts
import { NextRequest, NextResponse } from "next/server";
import { getMoleculeImage, RasynError } from "@/lib/rasyn-client";

export async function GET(request: NextRequest) {
  const smiles = request.nextUrl.searchParams.get("smiles");
  if (!smiles) return new NextResponse("Missing smiles param", { status: 400 });

  try {
    const svgBuffer = await getMoleculeImage(smiles);
    return new NextResponse(svgBuffer, {
      headers: {
        "Content-Type": "image/svg+xml",
        "Cache-Control": "public, max-age=86400",  // Cache SVGs for 1 day
      },
    });
  } catch (err) {
    return new NextResponse("Invalid SMILES", { status: 400 });
  }
}
```

### `hooks/useRetrosynthesis.ts` — React Hook

```typescript
// hooks/useRetrosynthesis.ts
"use client";

import { useState, useCallback } from "react";

interface Prediction {
  rank: number;
  reactants_smiles: string[];
  confidence: number;
  model_source: string;
  verification: {
    rdkit_valid: boolean;
    forward_match_score: number;
    overall_confidence: number;
  } | null;
}

interface SingleStepResult {
  product: string;
  predictions: Prediction[];
  compute_time_ms: number;
  error: string | null;
}

interface Route {
  steps: Array<{
    product: string;
    reactants: string[];
    confidence: number;
  }>;
  total_score: number;
  num_steps: number;
  all_available: boolean;
  starting_materials: string[];
}

interface MultiStepResult {
  target: string;
  routes: Route[];
  compute_time_ms: number;
  error: string | null;
}

export function useRetrosynthesis() {
  const [loading, setLoading] = useState(false);
  const [singleStepResult, setSingleStepResult] = useState<SingleStepResult | null>(null);
  const [multiStepResult, setMultiStepResult] = useState<MultiStepResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSingleStep = useCallback(
    async (smiles: string, model: string = "retro", topK: number = 10) => {
      setLoading(true);
      setError(null);
      setSingleStepResult(null);

      try {
        const res = await fetch("/api/retro/single-step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles, model, topK }),
        });

        if (!res.ok) {
          const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
          throw new Error(data.message || data.error || `Request failed (${res.status})`);
        }

        const data: SingleStepResult = await res.json();
        setSingleStepResult(data);
        if (data.error) setError(data.error);
      } catch (err: any) {
        setError(err.message || "Unknown error");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const runMultiStep = useCallback(
    async (smiles: string, maxDepth: number = 10, maxRoutes: number = 5) => {
      setLoading(true);
      setError(null);
      setMultiStepResult(null);

      try {
        const res = await fetch("/api/retro/multi-step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles, maxDepth, maxRoutes }),
        });

        if (!res.ok) {
          const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
          throw new Error(data.message || data.error || `Request failed (${res.status})`);
        }

        const data: MultiStepResult = await res.json();
        setMultiStepResult(data);
        if (data.error) setError(data.error);
      } catch (err: any) {
        setError(err.message || "Unknown error");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return {
    loading,
    error,
    singleStepResult,
    multiStepResult,
    runSingleStep,
    runMultiStep,
  };
}
```

### `hooks/useMoleculeValidation.ts`

```typescript
// hooks/useMoleculeValidation.ts
"use client";

import { useState, useCallback } from "react";

interface MoleculeInfo {
  valid: boolean;
  canonical: string | null;
  formula: string | null;
  mol_weight: number | null;
  svg: string | null;
}

export function useMoleculeValidation() {
  const [info, setInfo] = useState<MoleculeInfo | null>(null);
  const [validating, setValidating] = useState(false);

  const validate = useCallback(async (smiles: string) => {
    if (!smiles.trim()) {
      setInfo(null);
      return;
    }

    setValidating(true);
    try {
      const res = await fetch("/api/molecules/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: smiles.trim() }),
      });
      const data: MoleculeInfo = await res.json();
      setInfo(data);
    } catch {
      setInfo(null);
    } finally {
      setValidating(false);
    }
  }, []);

  return { info, validating, validate };
}
```

### `components/MoleculeImage.tsx`

```tsx
// components/MoleculeImage.tsx
"use client";

export function MoleculeImage({
  svg,
  size = "w-48 h-48",
}: {
  svg: string | null;
  size?: string;
}) {
  if (!svg) return null;
  return (
    <img
      src={`data:image/svg+xml;base64,${svg}`}
      alt="Molecule structure"
      className={size}
    />
  );
}
```

### `components/RetroSearch.tsx` — Complete Example

```tsx
// components/RetroSearch.tsx
"use client";

import { useState, useEffect } from "react";
import { useRetrosynthesis } from "@/hooks/useRetrosynthesis";
import { useMoleculeValidation } from "@/hooks/useMoleculeValidation";
import { MoleculeImage } from "@/components/MoleculeImage";

export function RetroSearch() {
  const [smiles, setSmiles] = useState("");
  const [model, setModel] = useState("retro");
  const [topK, setTopK] = useState(10);

  const { runSingleStep, loading, singleStepResult, error } = useRetrosynthesis();
  const { info: molInfo, validate } = useMoleculeValidation();

  // Validate SMILES as user types (debounced)
  useEffect(() => {
    const timer = setTimeout(() => {
      if (smiles.trim()) validate(smiles);
    }, 500);  // 500ms debounce
    return () => clearTimeout(timer);
  }, [smiles, validate]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (smiles.trim() && !loading) {
      runSingleStep(smiles.trim(), model, topK);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-bold">Retrosynthesis Analysis</h1>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Product SMILES</label>
          <input
            type="text"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (Aspirin)"
            className="w-full border rounded-lg px-4 py-2 text-lg font-mono"
          />
        </div>

        {/* Live molecule preview */}
        {molInfo && (
          <div className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
            <MoleculeImage svg={molInfo.svg} size="w-24 h-24" />
            <div className="text-sm">
              {molInfo.valid ? (
                <>
                  <p className="text-green-600 font-medium">Valid molecule</p>
                  <p>Formula: {molInfo.formula}</p>
                  <p>MW: {molInfo.mol_weight?.toFixed(1)} g/mol</p>
                </>
              ) : (
                <p className="text-red-600">Invalid SMILES notation</p>
              )}
            </div>
          </div>
        )}

        <div className="flex gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="border rounded px-3 py-2"
            >
              <option value="retro">RetroTransformer (fast)</option>
              <option value="llm">LLM (detailed, slower)</option>
              <option value="both">Both models</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Results</label>
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value) || 10)}
              min={1}
              max={50}
              className="border rounded px-3 py-2 w-20"
            />
          </div>
          <div className="flex items-end">
            <button
              type="submit"
              disabled={loading || !smiles.trim() || (molInfo !== null && !molInfo.valid)}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg font-medium disabled:opacity-50 hover:bg-blue-700"
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>
        </div>
      </form>

      {/* Loading state */}
      {loading && (
        <div className="text-center py-8">
          <div className="animate-spin w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full mx-auto" />
          <p className="mt-3 text-gray-500">
            {model === "llm"
              ? "Running LLM analysis... (may take 1-2 minutes on first use)"
              : "Analyzing molecule..."}
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg">
          {error}
        </div>
      )}

      {/* Results */}
      {singleStepResult && singleStepResult.predictions.length > 0 && (
        <div className="space-y-4">
          <p className="text-sm text-gray-500">
            {singleStepResult.predictions.length} predictions in{" "}
            {(singleStepResult.compute_time_ms / 1000).toFixed(1)}s
          </p>

          {singleStepResult.predictions.map((pred) => (
            <div key={pred.rank} className="border rounded-lg p-4 hover:bg-gray-50">
              <div className="flex justify-between items-start">
                <div>
                  <span className="font-bold text-lg">#{pred.rank}</span>
                  <span className="ml-3 text-sm text-gray-500">
                    via {pred.model_source}
                  </span>
                </div>
                <span className={`text-lg font-semibold ${
                  pred.confidence > 0.7 ? "text-green-600" :
                  pred.confidence > 0.3 ? "text-yellow-600" :
                  "text-red-600"
                }`}>
                  {(pred.confidence * 100).toFixed(1)}%
                </span>
              </div>

              <div className="mt-3 font-mono text-sm bg-gray-100 p-2 rounded">
                {pred.reactants_smiles.join(" + ")}
              </div>

              {pred.verification && (
                <div className="mt-2 text-xs text-gray-500 flex gap-4">
                  <span>
                    Chemically valid: {pred.verification.rdkit_valid ? "Yes" : "No"}
                  </span>
                  <span>
                    Forward verification: {(pred.verification.forward_match_score * 100).toFixed(0)}%
                  </span>
                </div>
              )}

              {/* Show molecule images for reactants */}
              <div className="mt-3 flex gap-2 flex-wrap">
                {pred.reactants_smiles.map((r, i) => (
                  <img
                    key={i}
                    src={`/api/molecules/image?smiles=${encodeURIComponent(r)}`}
                    alt={r}
                    className="w-24 h-24 border rounded"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {singleStepResult && singleStepResult.predictions.length === 0 && !error && (
        <div className="text-center py-8 text-gray-500">
          No predictions found for this molecule.
        </div>
      )}
    </div>
  );
}
```

---

## 10. Gradio Demo — Authentication & Access

A built-in interactive demo is available at `/demo`. This is useful for testing, internal demos, and showing investors.

### Access URL

```
http://rasyn-prod-alb-449951296.us-east-1.elb.amazonaws.com/demo/
```

After DNS setup:
```
https://api.rasyn.ai/demo/
```

### How Demo Authentication Works

The demo is protected by a **cookie-based login page**:

1. User visits `/demo/` → sees a login page asking for an "Access Token"
2. User enters either:
   - The **demo password**: `rasyn2026`
   - Any valid **API key** (e.g., `rsy_kA4GsJ4kR7Y3GmnndrOl_...`)
3. If valid → a signed session cookie is set (7 days), user is redirected to `/demo/`
4. All subsequent Gradio internal calls use the cookie automatically
5. If invalid → login page shows an error message

### Why Cookie-Based?

Gradio makes 10-20+ internal HTTP requests per page load (SSE connections, queue polling, etc.). These internal calls can't carry API keys in headers. The cookie is set once and included automatically on all requests.

### Sharing Demo Access

- **For investors/demos**: Share the URL and tell them the password `rasyn2026`
- **For developers**: They can use their API key as the token
- **Direct link with token**: `https://api.rasyn.ai/demo/?token=rasyn2026` (auto-authenticates and redirects)

### Changing the Demo Password

```bash
ssh -i ~/.ssh/rasyn-key.pem ubuntu@44.192.125.177
sudo systemctl edit rasyn
# Add under [Service]:
# Environment=RASYN_DEMO_PASS=your_new_password
sudo systemctl restart rasyn
```

### Demo Features

The demo has three tabs:
1. **Single-Step Retrosynthesis** — Enter SMILES, select model (LLM/RetroTx/Both), see ranked predictions
2. **Multi-Step Planning** — Enter target SMILES, get full route trees
3. **Molecule Validator** — Quick SMILES validation + structure preview

---

## 11. User Onboarding & Subscription Flow

Here's how to implement user signup → API key creation → gated access in your frontend:

### Architecture

```
User signs up on rasyn.ai
    ↓
Your backend creates a Rasyn API key (POST /api/v1/keys)
    ↓
Key stored in YOUR database (encrypted), linked to user account
    ↓
User makes retrosynthesis request
    ↓
Your backend checks: user authenticated? has credits/subscription?
    ↓
If yes → proxy request to Rasyn API using user's key
    ↓
Deduct credit, return result to browser
```

### Database Schema (your app)

```sql
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  plan TEXT DEFAULT 'free',    -- 'free', 'pro', 'enterprise'
  credits_remaining INT DEFAULT 10,
  rasyn_key_id TEXT,            -- Rasyn key ID (for revocation)
  rasyn_api_key TEXT,           -- Encrypted Rasyn API key
  created_at TIMESTAMP DEFAULT NOW()
);
```

### Complete Signup Flow

```typescript
// app/api/auth/signup/route.ts (Next.js)
import { NextRequest, NextResponse } from "next/server";
import { createRasynKey } from "@/lib/rasyn-admin";
import { db } from "@/lib/database";
import { encrypt } from "@/lib/crypto";

export async function POST(request: NextRequest) {
  const { email, password, name } = await request.json();

  // 1. Create user in your database
  const userId = crypto.randomUUID();
  // ... hash password, create user record ...

  // 2. Create a Rasyn API key for this user
  const rasynKey = await createRasynKey(email);

  // 3. Store the key (encrypted!) in your database
  await db.query(
    `INSERT INTO users (id, email, name, rasyn_key_id, rasyn_api_key, credits_remaining)
     VALUES ($1, $2, $3, $4, $5, $6)`,
    [userId, email, name, rasynKey.id, encrypt(rasynKey.key), 10]  // 10 free credits
  );

  // 4. Return success (never send the Rasyn key to the browser!)
  return NextResponse.json({
    success: true,
    user: { id: userId, email, name, credits: 10 },
  });
}
```

### Gated API Route

```typescript
// app/api/retro/single-step/route.ts
import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth";    // Your auth system
import { db } from "@/lib/database";
import { decrypt } from "@/lib/crypto";
import { singleStep, RasynError } from "@/lib/rasyn-client";

export async function POST(request: NextRequest) {
  // 1. Check user is logged in
  const session = await getSession(request);
  if (!session) {
    return NextResponse.json({ error: "Please log in" }, { status: 401 });
  }

  // 2. Check user has credits
  const user = await db.query("SELECT * FROM users WHERE id = $1", [session.userId]);
  if (!user) return NextResponse.json({ error: "User not found" }, { status: 404 });

  if (user.credits_remaining <= 0 && user.plan === "free") {
    return NextResponse.json(
      {
        error: "no_credits",
        message: "You've used all your free credits. Upgrade to Pro for unlimited access.",
        upgrade_url: "/pricing",
      },
      { status: 402 }
    );
  }

  // 3. Parse request
  const body = await request.json();
  const { smiles, model = "retro", topK = 10 } = body;

  if (!smiles) {
    return NextResponse.json({ error: "SMILES is required" }, { status: 400 });
  }

  // 4. Call Rasyn API with user's key
  try {
    const userRasynKey = decrypt(user.rasyn_api_key);
    const result = await singleStep(smiles, model, topK, true, userRasynKey);

    // 5. Deduct credit (skip for paid plans)
    if (user.plan === "free") {
      await db.query(
        "UPDATE users SET credits_remaining = credits_remaining - 1 WHERE id = $1",
        [session.userId]
      );
    }

    return NextResponse.json({
      ...result,
      credits_remaining: user.plan === "free" ? user.credits_remaining - 1 : null,
    });
  } catch (err) {
    if (err instanceof RasynError) {
      return NextResponse.json(
        { error: err.code, message: err.message },
        { status: err.status }
      );
    }
    console.error("Retro error:", err);
    return NextResponse.json({ error: "Service unavailable" }, { status: 503 });
  }
}
```

### Admin Dashboard — Usage Monitoring

```typescript
// app/api/admin/usage/route.ts
import { NextRequest, NextResponse } from "next/server";
import { listRasynKeys } from "@/lib/rasyn-admin";
import { requireAdmin } from "@/lib/auth";

export async function GET(request: NextRequest) {
  await requireAdmin(request);

  // Get usage stats from Rasyn API
  const { keys, total } = await listRasynKeys();

  // Format for your admin dashboard
  const usage = keys.map((k) => ({
    keyId: k.id,
    userName: k.name.replace("user-", ""),
    totalRequests: k.request_count,
    lastUsed: k.last_used_at,
    isActive: k.is_active === 1,
  }));

  return NextResponse.json({ usage, totalKeys: total });
}
```

### Handling Key Revocation (User Cancels Subscription)

```typescript
// When user cancels or you need to revoke access:
import { revokeRasynKey } from "@/lib/rasyn-admin";

async function cancelUserSubscription(userId: string) {
  const user = await db.query("SELECT rasyn_key_id FROM users WHERE id = $1", [userId]);

  if (user.rasyn_key_id) {
    // Revoke their Rasyn API key (stops working within 30s)
    await revokeRasynKey(user.rasyn_key_id);

    // Clear from your DB
    await db.query(
      "UPDATE users SET rasyn_key_id = NULL, rasyn_api_key = NULL WHERE id = $1",
      [userId]
    );
  }
}
```

---

## 12. Performance, Timeouts & UX

### Response Time Expectations

| Endpoint | First Call | Subsequent Calls | Notes |
|----------|-----------|------------------|-------|
| Validate SMILES | <100ms | <50ms | No GPU needed |
| Molecule Image | <200ms | <100ms | No GPU needed |
| Single-step (retro) | 1–3s | 0.2–1s | RetroTransformer |
| Single-step (llm) | 30–200s | 3–10s | First call loads 6.4GB model to GPU |
| Single-step (both) | 30–200s | 3–10s | Runs both models |
| Multi-step | 10–120s | 10–60s | A* search, many single-step calls |
| Health check | <50ms | <50ms | No GPU needed |

### Why "First Call" is So Slow

Models are **lazy-loaded** — they load into GPU memory only when first requested:
- RetroTransformer v2: ~500MB, loads in ~5s
- LLM (RSGPT v6): ~6.4GB, loads in ~30-120s
- Forward model: ~500MB, loads in ~5s

After the first call, subsequent calls are fast because models stay in GPU memory.

### Recommended Client Timeouts

```typescript
const TIMEOUTS_MS = {
  validate:       10_000,   // 10 seconds
  image:          10_000,   // 10 seconds
  singleRetro:    30_000,   // 30 seconds
  singleLLM:     300_000,   // 5 minutes (first call loads model)
  singleBoth:    300_000,   // 5 minutes
  multiStep:     600_000,   // 10 minutes
  health:          5_000,   // 5 seconds
};
```

### UX Recommendations

| Scenario | What to Show |
|----------|-------------|
| Validating SMILES | Small spinner next to input, with molecule preview on success |
| Single-step (retro) | Spinner: "Analyzing molecule... (typically 1-5 seconds)" |
| Single-step (llm, first time) | Spinner: "Loading AI model... (this may take 1-2 minutes on first use)" |
| Single-step (llm, subsequent) | Spinner: "Running detailed analysis... (typically 3-10 seconds)" |
| Multi-step | Progress bar or step counter: "Planning synthesis route... (typically 10-60 seconds)" |
| Rate limited | Toast: "Too many requests. Please wait X seconds." with countdown |
| Timeout | "The analysis took too long. Try the RetroTransformer model for faster results." |

### Pre-warming Models

You can pre-warm models on server start to avoid the slow first call:

```bash
# After deploying or restarting the service:
curl -X POST https://api.rasyn.ai/api/v1/retro/single-step \
  -H "Content-Type: application/json" \
  -H "X-API-Key: rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq" \
  -d '{"smiles": "c1ccccc1", "model": "both", "top_k": 1}'
```

This loads all models into GPU memory. Subsequent user calls will be fast.

---

## 13. Security Checklist — Before Going Live

### DNS & HTTPS (must do)

- [ ] Add ACM validation CNAME record (Section 2, Step 1)
- [ ] Wait for cert to show `ISSUED` in ACM console
- [ ] Add `api.rasyn.ai` CNAME record (Section 2, Step 2)
- [ ] Create HTTPS listener on ALB (Section 2, Step 3)
- [ ] Redirect HTTP → HTTPS (Section 2, Step 4)
- [ ] Verify: `curl https://api.rasyn.ai/api/v1/health` returns `{"status":"ok",...}`

### API Keys (must do)

- [ ] Store `RASYN_ADMIN_KEY` in server-side `.env.local` — NEVER in client code
- [ ] Add `.env.local` to `.gitignore`
- [ ] Implement per-user API keys (Section 11) OR use shared key with your own auth

### Your App's Own Auth (must do)

- [ ] Implement user authentication (NextAuth, Clerk, Auth0, etc.)
- [ ] Gate all `/api/*` proxy routes behind your auth (check session/JWT)
- [ ] Implement credit/subscription checks before proxying to Rasyn

### Nice to Have

- [ ] Encrypt stored Rasyn API keys in your database
- [ ] Implement key rotation (periodically create new keys, revoke old ones)
- [ ] Set up monitoring — check `request_count` via `GET /api/v1/keys` periodically
- [ ] Rate-limit your own API routes too (prevent individual users from hammering the backend)
- [ ] Ask Ansh to restrict SSH access to specific IPs (`AllowedSSHCidr` in CloudFormation)
- [ ] Check CloudWatch logs at `/rasyn/rasyn-prod` for errors

---

## 14. AWS Infrastructure Reference

| Resource | Value |
|----------|-------|
| EC2 Instance | `i-0971f214c77aec714` (g5.xlarge, NVIDIA A10G 24GB) |
| Instance IP | `44.192.125.177` |
| ALB DNS | `rasyn-prod-alb-449951296.us-east-1.elb.amazonaws.com` |
| Target Group | `rasyn-prod-tg` |
| S3 Model Bucket | `s3://rasyn-models-779530687716` |
| ACM Certificate | `arn:aws:acm:us-east-1:779530687716:certificate/9e0f3fef-6e47-4896-bc9b-72d01c3eef5e` |
| CloudWatch Logs | `/rasyn/rasyn-prod` |
| Region | `us-east-1` |
| SSH | `ssh -i ~/.ssh/rasyn-key.pem ubuntu@44.192.125.177` |
| App Directory | `/opt/rasyn/app` |
| Python venv | `/opt/rasyn/venv` |
| API Keys DB | `/opt/rasyn/data/api_keys.db` |
| Systemd Service | `rasyn.service` |

### Server Management Commands

```bash
# SSH into the server
ssh -i ~/.ssh/rasyn-key.pem ubuntu@44.192.125.177

# Check service status
sudo systemctl status rasyn

# View live logs
sudo journalctl -u rasyn -f

# Restart the service
sudo systemctl restart rasyn

# Edit environment variables
sudo systemctl edit rasyn
# Then: sudo systemctl daemon-reload && sudo systemctl restart rasyn

# Current environment variables set:
# RASYN_API_KEYS=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq
# RASYN_DEMO_PASS=rasyn2026
```

---

## 15. cURL Examples — Quick Testing

```bash
# Use this base URL until DNS is set up:
BASE="http://rasyn-prod-alb-449951296.us-east-1.elb.amazonaws.com"
KEY="rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq"

# After DNS setup, use:
# BASE="https://api.rasyn.ai"

# --- Health check (no auth) ---
curl $BASE/api/v1/health

# --- Single-step retrosynthesis (Aspirin) ---
curl -X POST $BASE/api/v1/retro/single-step \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "model": "retro", "top_k": 5}'

# --- Multi-step route planning ---
curl -X POST $BASE/api/v1/retro/multi-step \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "max_depth": 10, "max_routes": 3}'

# --- Validate SMILES ---
curl -X POST $BASE/api/v1/molecules/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"smiles": "c1ccccc1"}'

# --- Molecule image (SVG) ---
curl "$BASE/api/v1/molecules/image?smiles=c1ccccc1&api_key=$KEY" -o benzene.svg

# --- Create a new API key (admin only) ---
curl -X POST $BASE/api/v1/keys \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"name": "test-user@example.com", "role": "user"}'

# --- List all API keys (admin only) ---
curl $BASE/api/v1/keys \
  -H "X-API-Key: $KEY"

# --- Revoke a key (admin only) ---
curl -X DELETE $BASE/api/v1/keys/KEY_ID_HERE \
  -H "X-API-Key: $KEY"

# --- Access the demo (opens login page) ---
# In browser: $BASE/demo/
# Or with auto-login: $BASE/demo/?token=rasyn2026
```

---

## 16. FAQ & Troubleshooting

### Q: I'm getting CORS errors in the browser

**A:** You should NOT be calling the Rasyn API directly from the browser. All calls must go through your Next.js API routes (server-side). Server-to-server calls don't have CORS restrictions. See Section 9.

### Q: The first API call takes forever (30+ seconds)

**A:** This is expected — AI models are lazy-loaded into GPU memory on first use. The LLM is 6.4GB and takes 30-120s to load. Subsequent calls are fast (1-10s). You can pre-warm models after deploy (see Section 12).

### Q: I got a 429 rate limit error

**A:** You've exceeded the rate limit for that endpoint. Check the `Retry-After` header for how long to wait. For multi-step, the limit is only 5 requests per minute because it's very GPU-intensive.

### Q: My API key stopped working

**A:** The key may have been revoked. Check with Ansh or use the admin key to `GET /api/v1/keys` and check `is_active`. Create a new key if needed.

### Q: How do I test without the real API?

**A:** You can mock the API responses in your Next.js API routes during development:

```typescript
// In development, return mock data:
if (process.env.NODE_ENV === "development" && process.env.MOCK_RASYN === "true") {
  return NextResponse.json({
    product: smiles,
    predictions: [
      {
        rank: 1,
        reactants_smiles: ["CCO", "CC(=O)Cl"],
        confidence: 0.85,
        model_source: "retro_v2",
        verification: null,
        edit_info: null,
      },
    ],
    compute_time_ms: 123,
    error: null,
  });
}
```

### Q: The demo login page won't accept my password

**A:** The default demo password is `rasyn2026`. You can also use any valid API key. If it still doesn't work, the service may need restarting — contact Ansh.

### Q: How do I add a molecule drawing editor (like ChemDraw)?

**A:** Consider integrating [Ketcher](https://github.com/epam/ketcher) (open-source, React-compatible) or [JSME](https://peter-ertl.com/jsme/) (free for academic use). These editors output SMILES strings that you can feed directly into the Rasyn API.

### Q: What happens if the GPU runs out of memory?

**A:** The API returns a 500 error with `"CUDA out of memory"` in the error message. This can happen with very large molecules. Reducing `top_k` or using `"retro"` model (smaller) helps. Contact Ansh if this happens frequently.

### Q: Can I call the API from Python instead of JavaScript?

**A:** Yes:

```python
import requests

BASE = "https://api.rasyn.ai"
KEY = "rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq"

response = requests.post(
    f"{BASE}/api/v1/retro/single-step",
    json={"smiles": "CC(=O)Oc1ccccc1C(=O)O", "model": "retro", "top_k": 5},
    headers={"X-API-Key": KEY},
    timeout=300,
)
print(response.json())
```

---

## Summary — What the Developer Needs To Do

1. **DNS** (10 min): Add 2 CNAME records in your DNS provider (Section 2)
2. **Tell Ansh** when cert is issued so he can create the HTTPS listener
3. **Create Next.js API routes** that proxy to Rasyn API with the key (Section 9)
4. **Build your auth system** — gate the proxy routes behind login (Section 11)
5. **Implement user signup flow** — create per-user Rasyn keys on signup (Section 11)
6. **Implement credits/billing** — check and deduct credits before proxying (Section 11)
7. **Build the UI** — use the React hooks and components (Section 9)
8. **Test** with cURL first (Section 15), then through the UI

**Questions?** Contact Ansh for backend/API issues.
