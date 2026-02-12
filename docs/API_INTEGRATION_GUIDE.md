# Rasyn API Integration Guide

> **For**: Frontend developer wiring rasyn.ai to the Rasyn Retrosynthesis API
> **Version**: 1.0.0
> **Last updated**: 2026-02-12

---

## Table of Contents

1. [Overview](#1-overview)
2. [Base URL & Environments](#2-base-url--environments)
3. [Authentication](#3-authentication)
4. [HTTPS & DNS Setup (REQUIRED)](#4-https--dns-setup-required)
5. [CORS Configuration](#5-cors-configuration)
6. [Rate Limits](#6-rate-limits)
7. [API Endpoints Reference](#7-api-endpoints-reference)
   - [Health Check](#71-health-check)
   - [Single-Step Retrosynthesis](#72-single-step-retrosynthesis)
   - [Multi-Step Route Planning](#73-multi-step-route-planning)
   - [Validate SMILES](#74-validate-smiles)
   - [Molecule Image](#75-molecule-image)
8. [Error Handling](#8-error-handling)
9. [Frontend Integration Guide (React/Next.js)](#9-frontend-integration-guide-reactnextjs)
10. [Gradio Demo](#10-gradio-demo)
11. [Performance & Timeouts](#11-performance--timeouts)
12. [Security Checklist](#12-security-checklist)

---

## 1. Overview

Rasyn provides AI-powered retrosynthetic analysis through a REST API. Given a target molecule (as a SMILES string), the API predicts how to synthesize it — either one step back (single-step) or a full route to purchasable starting materials (multi-step).

**Two AI models are available:**
- **LLM (RSGPT)** — Large language model fine-tuned on reaction data. Higher quality but slower (~30-200s first call, ~3-10s subsequent). Includes round-trip verification.
- **RetroTransformer v2** — Specialized seq2seq model. Faster (~0.2-1s) but no verification.

**Architecture:**
```
Browser/App  →  ALB (HTTPS)  →  EC2 GPU Instance  →  FastAPI + Models on GPU
                                                      ├── LLM (RSGPT v6)
                                                      ├── RetroTransformer v2
                                                      ├── Forward Model (verification)
                                                      └── Graph Edit Head
```

---

## 2. Base URL & Environments

| Environment | Base URL |
|-------------|----------|
| Production  | `https://api.rasyn.ai` (after DNS setup — see Section 4) |
| Direct ALB  | `http://rasyn-prod-alb-449951296.us-east-1.elb.amazonaws.com` |
| Local dev   | `http://localhost:8000` |

> **Important**: The direct ALB URL is HTTP only and temporary. Once DNS is configured, all traffic should go through `https://api.rasyn.ai`.

---

## 3. Authentication

All API endpoints (except `/api/v1/health`) require authentication via API key.

### Passing the API Key

Three methods are supported (use whichever fits your architecture):

#### Option A: `X-API-Key` Header (Recommended for backend-to-API calls)
```http
POST /api/v1/retro/single-step HTTP/1.1
Host: api.rasyn.ai
Content-Type: application/json
X-API-Key: rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq

{"smiles": "CC(=O)Oc1ccccc1C(O)=O"}
```

#### Option B: `Authorization: Bearer` Header (Standard OAuth-style)
```http
POST /api/v1/retro/single-step HTTP/1.1
Host: api.rasyn.ai
Content-Type: application/json
Authorization: Bearer rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq

{"smiles": "CC(=O)Oc1ccccc1C(O)=O"}
```

#### Option C: Query Parameter (For browser/Gradio compatibility)
```
GET /api/v1/molecules/image?smiles=c1ccccc1&api_key=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq
```

### Current API Key

```
rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq
```

> **CRITICAL: Never expose this key in client-side JavaScript!** If the frontend is a SPA (React/Next.js), the API calls must go through YOUR backend server (Next.js API routes, Express, etc.) which adds the key server-side. See [Section 9](#9-frontend-integration-guide-reactnextjs) for the pattern.

### Auth Error Responses

**401 — No key provided:**
```json
{
  "error": "authentication_required",
  "message": "API key required. Pass via X-API-Key header, Authorization: Bearer <key>, or ?api_key= query param."
}
```

**403 — Invalid key:**
```json
{
  "error": "invalid_api_key",
  "message": "The provided API key is not valid."
}
```

---

## 4. HTTPS & DNS Setup (REQUIRED)

An SSL certificate has been requested via AWS ACM for `api.rasyn.ai`. **You must add the following DNS record to validate it:**

### Step 1: Add DNS CNAME for Certificate Validation

| Type  | Name | Value |
|-------|------|-------|
| CNAME | `_21f7ed329fffff89e760e0a1716a6f1e.api.rasyn.ai` | `_a14be9ea59fed95b98c5ab37ac49aed6.jkddzztszm.acm-validations.aws.` |

Add this record in your DNS provider (wherever rasyn.ai is managed). AWS will automatically validate and issue the certificate once this record propagates (usually 5-30 minutes).

### Step 2: Point api.rasyn.ai to the ALB

After the certificate is issued, add a CNAME record for the API subdomain:

| Type  | Name | Value |
|-------|------|-------|
| CNAME | `api.rasyn.ai` | `rasyn-prod-alb-449951296.us-east-1.elb.amazonaws.com` |

### Step 3: Create the HTTPS Listener

Once the certificate status is `ISSUED` (check in AWS Console > ACM), run this command (or ask Ansh to run it):

```bash
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:779530687716:loadbalancer/app/rasyn-prod-alb/53d5b0dd8df7ab1f \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:us-east-1:779530687716:certificate/9e0f3fef-6e47-4896-bc9b-72d01c3eef5e \
  --default-actions "Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:779530687716:targetgroup/rasyn-prod-tg/20b35741cfb1023c" \
  --region us-east-1
```

### Step 4: Redirect HTTP to HTTPS

Modify the existing HTTP listener to redirect:

```bash
# Get the HTTP listener ARN
aws elbv2 describe-listeners \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:779530687716:loadbalancer/app/rasyn-prod-alb/53d5b0dd8df7ab1f \
  --query "Listeners[?Port==\`80\`].ListenerArn" --output text --region us-east-1

# Modify it to redirect
aws elbv2 modify-listener \
  --listener-arn <HTTP_LISTENER_ARN> \
  --default-actions "Type=redirect,RedirectConfig={Protocol=HTTPS,Port=443,StatusCode=HTTP_301}" \
  --region us-east-1
```

---

## 5. CORS Configuration

The API allows requests from these origins:

```
https://rasyn.ai
https://www.rasyn.ai
https://app.rasyn.ai
http://localhost:3000   (for local development)
```

**Allowed methods:** `GET`, `POST`, `OPTIONS`
**Allowed headers:** `Content-Type`, `Authorization`, `X-API-Key`
**Exposed headers:** `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

To add more origins (e.g., staging), set the `RASYN_CORS_ORIGINS` environment variable on the EC2 instance:

```bash
# SSH into EC2
ssh -i ~/.ssh/rasyn-key.pem ubuntu@44.192.125.177

# Edit systemd service
sudo systemctl edit rasyn
# Add under [Service]:
# Environment=RASYN_CORS_ORIGINS=https://rasyn.ai,https://staging.rasyn.ai,http://localhost:3000

sudo systemctl restart rasyn
```

---

## 6. Rate Limits

Per-IP rate limiting is enforced to prevent abuse and protect GPU resources.

| Endpoint | Limit | Window |
|----------|-------|--------|
| `POST /api/v1/retro/single-step` | 20 requests | per minute |
| `POST /api/v1/retro/multi-step` | 5 requests | per minute |
| `POST /api/v1/molecules/*` | 60 requests | per minute |
| `GET /api/v1/molecules/*` | 60 requests | per minute |
| `/demo/*` | 30 requests | per minute |
| All other paths | 60 requests | per minute |
| `GET /api/v1/health` | Unlimited | — |

### Rate Limit Response Headers

Every response includes:
```
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 17
X-RateLimit-Reset: 1739395200
```

### 429 Too Many Requests

When the limit is exceeded:
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Max 20 requests per 60s for this endpoint.",
  "retry_after": "45"
}
```

**Frontend handling:** Show a user-friendly message and implement exponential backoff or queue requests.

---

## 7. API Endpoints Reference

### 7.1 Health Check

Check if the API is running and which models are loaded.

```
GET /api/v1/health
```

**No authentication required.**

**Response:**
```json
{
  "status": "ok",
  "models_loaded": ["retro_v2", "llm", "forward"],
  "device": "cuda"
}
```

Note: `models_loaded` only includes models that have been lazy-loaded (called at least once). On fresh startup, this will be `[]`.

---

### 7.2 Single-Step Retrosynthesis

Given a product molecule, predict what reactants could produce it (one synthetic step back).

```
POST /api/v1/retro/single-step
```

**Request Body:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "model": "retro",
  "top_k": 10,
  "use_verification": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `smiles` | string | **required** | Product molecule in SMILES notation |
| `model` | string | `"llm"` | Which model to use: `"llm"`, `"retro"`, or `"both"` |
| `top_k` | integer | `10` | Max predictions to return (1-50) |
| `use_verification` | boolean | `true` | Run forward-model verification (LLM only) |

**Model Selection Guide:**
| Model | Speed | Quality | Use When |
|-------|-------|---------|----------|
| `"retro"` | ~0.2-1s | Good | Quick results, interactive UI |
| `"llm"` | ~3-200s | Better (with verification) | Detailed analysis, first call is slow (model loading) |
| `"both"` | ~3-200s | Best (merged + ranked) | Full comparison |

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

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `product` | string | Canonical SMILES of the input product |
| `predictions` | array | Ranked list of predicted disconnections |
| `predictions[].rank` | integer | 1-indexed rank |
| `predictions[].reactants_smiles` | string[] | List of reactant SMILES strings |
| `predictions[].confidence` | float | Confidence score (0.0 to 1.0) |
| `predictions[].model_source` | string | `"llm"` or `"retro_v2"` |
| `predictions[].verification` | object\|null | Forward-model verification (LLM only) |
| `predictions[].verification.rdkit_valid` | boolean | Whether reactant SMILES are valid |
| `predictions[].verification.forward_match_score` | float | How well forward prediction matches the original product (0-1) |
| `predictions[].verification.overall_confidence` | float | Combined confidence score |
| `predictions[].edit_info` | object\|null | Bond edit information (when available) |
| `compute_time_ms` | float | Server-side compute time in milliseconds |
| `error` | string\|null | Error message if something went wrong |

---

### 7.3 Multi-Step Route Planning

Given a target molecule, plan a full synthetic route back to purchasable starting materials.

```
POST /api/v1/retro/multi-step
```

**Request Body:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "max_depth": 10,
  "max_routes": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `smiles` | string | **required** | Target molecule SMILES |
| `max_depth` | integer | `10` | Maximum retrosynthetic steps (1-20) |
| `max_routes` | integer | `5` | Maximum number of routes to return (1-20) |

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

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `target` | string | Canonical SMILES of the target |
| `routes` | array | List of complete synthetic routes, ranked by score |
| `routes[].steps` | array | Ordered list of retrosynthetic steps (target → starting materials) |
| `routes[].steps[].product` | string | Product of this step |
| `routes[].steps[].reactants` | string[] | Reactants for this step |
| `routes[].steps[].confidence` | float | Model confidence (0-1) |
| `routes[].total_score` | float | Aggregate route score (higher = better) |
| `routes[].num_steps` | integer | Number of steps in the route |
| `routes[].all_available` | boolean | Whether all starting materials are purchasable |
| `routes[].starting_materials` | string[] | Final starting materials (leaf nodes) |
| `compute_time_ms` | float | Server-side compute time in milliseconds |
| `error` | string\|null | Error message if something went wrong |

> **Warning:** Multi-step planning is computationally expensive. Typical response times are 10-60 seconds. Set client-side timeouts accordingly (see [Section 11](#11-performance--timeouts)).

---

### 7.4 Validate SMILES

Validate a SMILES string and get molecular information.

```
POST /api/v1/molecules/validate
```

**Request Body:**
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
  "svg": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `valid` | boolean | Whether the SMILES is chemically valid |
| `canonical` | string\|null | Canonicalized SMILES (null if invalid) |
| `formula` | string\|null | Molecular formula |
| `mol_weight` | float\|null | Molecular weight in g/mol |
| `svg` | string\|null | Base64-encoded SVG image of the molecule |

**Use this for:** Input validation before calling retrosynthesis, showing molecular structure previews, displaying molecular info in the UI.

---

### 7.5 Molecule Image

Get an SVG rendering of a molecule.

```
GET /api/v1/molecules/image?smiles=CC(=O)Oc1ccccc1C(=O)O
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `smiles` | string (query) | **Required.** SMILES string (URL-encoded) |

**Response:** SVG image (`Content-Type: image/svg+xml`)

**Use this for:** Inline molecule rendering in HTML:
```html
<img src="https://api.rasyn.ai/api/v1/molecules/image?smiles=c1ccccc1&api_key=YOUR_KEY" alt="Benzene" />
```

**Error:** Returns HTTP 400 with text `"Invalid SMILES"` if the SMILES is invalid.

---

## 8. Error Handling

### HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| 200 | Success | Request completed |
| 400 | Bad Request | Invalid SMILES, malformed JSON |
| 401 | Unauthorized | Missing API key |
| 403 | Forbidden | Invalid API key |
| 422 | Validation Error | Request body doesn't match schema |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Model error, GPU out of memory |

### 422 Validation Error (FastAPI/Pydantic)

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

### 500 Internal Server Error

When the model fails internally, the response body may still be the standard response format with an `error` field:

```json
{
  "product": "CC(=O)Oc1ccccc1C(=O)O",
  "predictions": [],
  "compute_time_ms": 0,
  "error": "Model inference failed: CUDA out of memory"
}
```

Or it may be a plain `"Internal Server Error"` string if the error occurs before the response is constructed.

### Recommended Frontend Error Handling

```typescript
async function callRasynAPI(endpoint: string, body: object) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.RASYN_API_KEY!,  // Server-side only!
    },
    body: JSON.stringify(body),
  });

  if (res.status === 401) throw new Error("API key missing or expired");
  if (res.status === 403) throw new Error("Invalid API key");
  if (res.status === 429) {
    const retryAfter = res.headers.get("Retry-After") || "60";
    throw new Error(`Rate limited. Retry in ${retryAfter}s`);
  }
  if (res.status === 422) {
    const detail = await res.json();
    throw new Error(`Validation error: ${JSON.stringify(detail)}`);
  }
  if (!res.ok) throw new Error(`API error: ${res.status}`);

  const data = await res.json();
  if (data.error) {
    console.warn("API returned error:", data.error);
  }
  return data;
}
```

---

## 9. Frontend Integration Guide (React/Next.js)

### Architecture: NEVER Expose the API Key to the Browser

```
Browser (React SPA)
    ↓ fetch("/api/retro/single-step", { smiles, model })
Your Backend (Next.js API Route / Express)
    ↓ fetch("https://api.rasyn.ai/api/v1/retro/single-step", { headers: { "X-API-Key": KEY } })
Rasyn API (EC2 GPU)
    ↓ returns predictions
Your Backend
    ↓ returns predictions
Browser
```

The API key lives ONLY on your backend server. The browser never sees it.

### Next.js API Route (App Router)

Create `app/api/retro/single-step/route.ts`:

```typescript
// app/api/retro/single-step/route.ts
import { NextRequest, NextResponse } from "next/server";

const RASYN_API_URL = process.env.RASYN_API_URL || "https://api.rasyn.ai";
const RASYN_API_KEY = process.env.RASYN_API_KEY!;

export async function POST(request: NextRequest) {
  // 1. Authenticate your user (e.g., check session/JWT)
  // const session = await getServerSession(authOptions);
  // if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  // 2. Parse request from frontend
  const body = await request.json();
  const { smiles, model = "retro", topK = 10 } = body;

  if (!smiles || typeof smiles !== "string") {
    return NextResponse.json({ error: "SMILES is required" }, { status: 400 });
  }

  // 3. Call Rasyn API with server-side API key
  try {
    const rasynRes = await fetch(`${RASYN_API_URL}/api/v1/retro/single-step`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": RASYN_API_KEY,
      },
      body: JSON.stringify({
        smiles,
        model,
        top_k: topK,
        use_verification: true,
      }),
      signal: AbortSignal.timeout(300_000), // 5 min timeout for LLM
    });

    if (!rasynRes.ok) {
      const errorText = await rasynRes.text();
      console.error("Rasyn API error:", rasynRes.status, errorText);
      return NextResponse.json(
        { error: "Retrosynthesis failed", detail: errorText },
        { status: rasynRes.status }
      );
    }

    const data = await rasynRes.json();

    // 4. (Optional) Check user's plan/credits before returning
    // await deductCredits(session.user.id, 1);

    return NextResponse.json(data);
  } catch (err: any) {
    if (err.name === "TimeoutError") {
      return NextResponse.json({ error: "Request timed out" }, { status: 504 });
    }
    console.error("Rasyn API call failed:", err);
    return NextResponse.json({ error: "Service unavailable" }, { status: 503 });
  }
}
```

### Next.js API Route: Multi-Step

Create `app/api/retro/multi-step/route.ts`:

```typescript
// app/api/retro/multi-step/route.ts
import { NextRequest, NextResponse } from "next/server";

const RASYN_API_URL = process.env.RASYN_API_URL || "https://api.rasyn.ai";
const RASYN_API_KEY = process.env.RASYN_API_KEY!;

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { smiles, maxDepth = 10, maxRoutes = 5 } = body;

  if (!smiles || typeof smiles !== "string") {
    return NextResponse.json({ error: "SMILES is required" }, { status: 400 });
  }

  try {
    const rasynRes = await fetch(`${RASYN_API_URL}/api/v1/retro/multi-step`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": RASYN_API_KEY,
      },
      body: JSON.stringify({
        smiles,
        max_depth: maxDepth,
        max_routes: maxRoutes,
      }),
      signal: AbortSignal.timeout(600_000), // 10 min timeout
    });

    if (!rasynRes.ok) {
      const errorText = await rasynRes.text();
      return NextResponse.json(
        { error: "Route planning failed", detail: errorText },
        { status: rasynRes.status }
      );
    }

    return NextResponse.json(await rasynRes.json());
  } catch (err: any) {
    if (err.name === "TimeoutError") {
      return NextResponse.json({ error: "Request timed out" }, { status: 504 });
    }
    return NextResponse.json({ error: "Service unavailable" }, { status: 503 });
  }
}
```

### Next.js API Route: Validate SMILES

Create `app/api/molecules/validate/route.ts`:

```typescript
// app/api/molecules/validate/route.ts
import { NextRequest, NextResponse } from "next/server";

const RASYN_API_URL = process.env.RASYN_API_URL || "https://api.rasyn.ai";
const RASYN_API_KEY = process.env.RASYN_API_KEY!;

export async function POST(request: NextRequest) {
  const { smiles } = await request.json();

  if (!smiles) {
    return NextResponse.json({ error: "SMILES required" }, { status: 400 });
  }

  const rasynRes = await fetch(`${RASYN_API_URL}/api/v1/molecules/validate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": RASYN_API_KEY,
    },
    body: JSON.stringify({ smiles }),
  });

  return NextResponse.json(await rasynRes.json(), { status: rasynRes.status });
}
```

### Frontend React Hook

```typescript
// hooks/useRetrosynthesis.ts
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

export function useRetrosynthesis() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SingleStepResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSingleStep = useCallback(
    async (smiles: string, model: string = "retro", topK: number = 10) => {
      setLoading(true);
      setError(null);
      setResult(null);

      try {
        const res = await fetch("/api/retro/single-step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles, model, topK }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.error || `HTTP ${res.status}`);
        }

        const data: SingleStepResult = await res.json();
        setResult(data);

        if (data.error) {
          setError(data.error);
        }
      } catch (err: any) {
        setError(err.message || "Unknown error");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return { runSingleStep, loading, result, error };
}
```

### Frontend Component Example

```tsx
// components/RetroSearch.tsx
"use client";

import { useState } from "react";
import { useRetrosynthesis } from "@/hooks/useRetrosynthesis";

export function RetroSearch() {
  const [smiles, setSmiles] = useState("");
  const [model, setModel] = useState("retro");
  const { runSingleStep, loading, result, error } = useRetrosynthesis();

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <input
          type="text"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          placeholder="Enter SMILES (e.g., CC(=O)Oc1ccccc1C(=O)O)"
          className="flex-1 border rounded px-3 py-2"
        />
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="retro">RetroTransformer (fast)</option>
          <option value="llm">LLM (detailed)</option>
          <option value="both">Both models</option>
        </select>
        <button
          onClick={() => runSingleStep(smiles, model)}
          disabled={loading || !smiles}
          className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 text-red-700 p-3 rounded">{error}</div>
      )}

      {result && (
        <div className="space-y-3">
          <p className="text-sm text-gray-500">
            Computed in {(result.compute_time_ms / 1000).toFixed(1)}s
          </p>
          {result.predictions.map((pred) => (
            <div key={pred.rank} className="border rounded p-4">
              <div className="flex justify-between">
                <span className="font-bold">#{pred.rank}</span>
                <span className="text-sm">
                  {(pred.confidence * 100).toFixed(1)}% confidence
                  ({pred.model_source})
                </span>
              </div>
              <div className="mt-2 font-mono text-sm">
                {pred.reactants_smiles.join(" + ")}
              </div>
              {pred.verification && (
                <div className="mt-1 text-xs text-gray-500">
                  Valid: {pred.verification.rdkit_valid ? "Yes" : "No"} |
                  Forward match: {(pred.verification.forward_match_score * 100).toFixed(0)}%
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

### Environment Variables (.env.local)

```env
# Your Next.js backend calls Rasyn with this key (NEVER expose to browser)
RASYN_API_URL=https://api.rasyn.ai
RASYN_API_KEY=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq
```

### Displaying Molecule SVG Images

The validate endpoint returns a base64-encoded SVG. To render it:

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

Or fetch the image endpoint directly (through your proxy):

```typescript
// app/api/molecules/image/route.ts
export async function GET(request: NextRequest) {
  const smiles = request.nextUrl.searchParams.get("smiles");
  if (!smiles) return NextResponse.json({ error: "Missing smiles" }, { status: 400 });

  const rasynRes = await fetch(
    `${RASYN_API_URL}/api/v1/molecules/image?smiles=${encodeURIComponent(smiles)}`,
    { headers: { "X-API-Key": RASYN_API_KEY } }
  );

  if (!rasynRes.ok) return new NextResponse("Invalid SMILES", { status: 400 });

  return new NextResponse(rasynRes.body, {
    headers: { "Content-Type": "image/svg+xml", "Cache-Control": "public, max-age=86400" },
  });
}
```

---

## 10. Gradio Demo

A built-in interactive demo is available at `/demo`. After DNS setup:

```
https://api.rasyn.ai/demo/?api_key=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq
```

This is a Gradio interface with three tabs:
1. **Single-Step Retrosynthesis** — Enter SMILES, select model, see ranked predictions
2. **Multi-Step Planning** — Enter target SMILES, see full route tree
3. **Molecule Validator** — Quick SMILES validation + structure preview

The demo uses the same API key as the REST endpoints. It's useful for testing and internal demos.

---

## 11. Performance & Timeouts

### Response Time Expectations

| Endpoint | First Call | Subsequent Calls |
|----------|-----------|------------------|
| Single-step (retro) | ~1-3s | ~0.2-1s |
| Single-step (llm) | ~30-200s (model loading) | ~3-10s |
| Single-step (both) | ~30-200s (model loading) | ~3-10s |
| Multi-step | ~10-120s | ~10-60s |
| Validate | <100ms | <50ms |
| Image | <200ms | <100ms |

### Why "First Call" is Slow

Models are **lazy-loaded** — they only load into GPU memory when first requested. The LLM (RSGPT) is 6.4GB and takes ~30-120s to load. After the first call, subsequent calls are fast.

### Recommended Client Timeouts

```typescript
const TIMEOUTS = {
  singleStepRetro: 30_000,   // 30s
  singleStepLLM:  300_000,   // 5 min (first call loads model)
  multiStep:      600_000,   // 10 min
  validate:       10_000,    // 10s
  image:          10_000,    // 10s
};
```

### UX Recommendations

- **Show a loading spinner** with estimated time: "Analyzing molecule... (typically 1-5 seconds)"
- **For LLM first call**: Show "Loading AI model... (this may take 1-2 minutes on first use)"
- **For multi-step**: Show progress or "Planning synthesis route... (typically 10-60 seconds)"
- **Pre-warm models** on backend startup by calling health + a dummy prediction during deployment

---

## 12. Security Checklist

Before going to production:

- [ ] **DNS**: Add ACM validation CNAME record (Section 4, Step 1)
- [ ] **DNS**: Point `api.rasyn.ai` CNAME to ALB (Section 4, Step 2)
- [ ] **HTTPS**: Create HTTPS listener once cert is issued (Section 4, Step 3)
- [ ] **HTTPS**: Redirect HTTP to HTTPS (Section 4, Step 4)
- [ ] **API Key**: Store `RASYN_API_KEY` in your backend's `.env` — never in client code
- [ ] **API Key Rotation**: Plan to generate new keys periodically
- [ ] **AWS IAM**: Create an IAM user with minimal permissions (currently using root keys — dangerous!)
- [ ] **SSH**: Restrict `AllowedSSHCidr` in CloudFormation to your office/VPN IP
- [ ] **Monitoring**: Check CloudWatch logs at `/rasyn/rasyn-prod` for errors
- [ ] **User Auth**: Implement your own user authentication in the Next.js API routes
- [ ] **Payment**: Gate API route calls based on user subscription tier

### Adding New API Keys

To add additional API keys (e.g., one per frontend environment):

```bash
# Generate a new key
python3 -c "import secrets; print(f'rsy_{secrets.token_urlsafe(36)}')"

# Add to the EC2 instance (comma-separated)
ssh -i ~/.ssh/rasyn-key.pem ubuntu@44.192.125.177
sudo systemctl edit rasyn
# Update: Environment=RASYN_API_KEYS=key1,key2,key3
sudo systemctl restart rasyn
```

### Revoking a Key

Remove it from the `RASYN_API_KEYS` list and restart the service. The old key will immediately stop working.

---

## Quick Reference: cURL Examples

```bash
# Health check (no auth)
curl https://api.rasyn.ai/api/v1/health

# Single-step retrosynthesis
curl -X POST https://api.rasyn.ai/api/v1/retro/single-step \
  -H "Content-Type: application/json" \
  -H "X-API-Key: rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "model": "retro", "top_k": 5}'

# Multi-step route planning
curl -X POST https://api.rasyn.ai/api/v1/retro/multi-step \
  -H "Content-Type: application/json" \
  -H "X-API-Key: rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "max_depth": 10, "max_routes": 3}'

# Validate SMILES
curl -X POST https://api.rasyn.ai/api/v1/molecules/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq" \
  -d '{"smiles": "c1ccccc1"}'

# Molecule image (SVG)
curl "https://api.rasyn.ai/api/v1/molecules/image?smiles=c1ccccc1&api_key=rsy_kA4GsJ4kR7Y3GmnndrOl_hs794VJSyiwX9REEotwR5MRdGpq" \
  -o molecule.svg
```

---

## AWS Infrastructure Reference

| Resource | Value |
|----------|-------|
| EC2 Instance | `i-0971f214c77aec714` (g5.xlarge, A10G 24GB) |
| Instance IP | `44.192.125.177` |
| ALB DNS | `rasyn-prod-alb-449951296.us-east-1.elb.amazonaws.com` |
| Target Group | `rasyn-prod-tg` |
| S3 Models | `s3://rasyn-models-779530687716` |
| ACM Cert ARN | `arn:aws:acm:us-east-1:779530687716:certificate/9e0f3fef-6e47-4896-bc9b-72d01c3eef5e` |
| CloudWatch Logs | `/rasyn/rasyn-prod` |
| SSH | `ssh -i ~/.ssh/rasyn-key.pem ubuntu@44.192.125.177` |
| Region | `us-east-1` |
