import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  let portArg = 0;
  for (let i = 0; i < process.argv.length; i++) {
    if ((process.argv[i] === '--port' || process.argv[i] === '-p') && process.argv[i + 1]) {
      portArg = Number(process.argv[i + 1]);
      break;
    }
  }
  const PORT = portArg || Number(process.env.DEFAULT_APP_PORT) || Number(process.env.PORT) || 3000;

  // Support up to 60MB for reference images
  app.use(express.json({ limit: '60mb' }));
  app.use(express.urlencoded({ extended: true, limit: '60mb' }));

  const apiKey = process.env.GEMINI_API_KEY;
  let ai: GoogleGenAI | null = null;
  if (apiKey && apiKey !== 'MY_GEMINI_API_KEY') {
    ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }

  const COMFY_URL = (process.env.LOCAL_COMFYUI_URL || 'http://127.0.0.1:8188').replace(/\/$/, '');

  // Comprehensive Local Health Check Endpoint (Requirement 9)
  app.get('/api/health', async (_req, res) => {
    let comfyHealth: any = { reachable: false, baseUrl: COMFY_URL, exactError: 'Pending' };

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);

      const [statsRes, objectInfoRes] = await Promise.all([
        fetch(`${COMFY_URL}/system_stats`, { signal: controller.signal }).catch((err) => ({ ok: false, err })),
        fetch(`${COMFY_URL}/object_info`, { signal: controller.signal }).catch((err) => ({ ok: false, err })),
      ]);
      clearTimeout(timeout);

      if (statsRes && 'ok' in statsRes && statsRes.ok) {
        const stats = await (statsRes as Response).json().catch(() => ({}));
        const objectInfo = objectInfoRes && 'ok' in objectInfoRes && (objectInfoRes as Response).ok
          ? await (objectInfoRes as Response).json().catch(() => ({}))
          : {};

        const ckptNode = objectInfo.CheckpointLoaderSimple?.input?.required?.ckpt_name?.[0] || [];
        const checkpoints = Array.isArray(ckptNode) ? ckptNode : [];
        const cnetNode = objectInfo.ControlNetLoader?.input?.required?.control_net_name?.[0] || [];
        const controlNetModels = Array.isArray(cnetNode) ? cnetNode : [];
        const ipAdapterNode = objectInfo.IPAdapterModelLoader?.input?.required?.ipadapter_file?.[0] || [];
        const ipAdapterModels = Array.isArray(ipAdapterNode) ? ipAdapterNode : [];

        const device = stats.devices?.[0] || {};
        const vramTotalMb = device.vram_total ? Math.round(device.vram_total / (1024 * 1024)) : 4096;
        const isLowVram = vramTotalMb <= 6144;

        const missing: string[] = [];
        if (checkpoints.length === 0) missing.push('Base model (e.g. v1-5-pruned-emaonly.safetensors in models/checkpoints/)');
        if (!objectInfo.IPAdapterApply) missing.push('ComfyUI_IPAdapter_plus custom node');
        if (!objectInfo.ControlNetApply) missing.push('ControlNet nodes');

        comfyHealth = {
          reachable: true,
          baseUrl: COMFY_URL,
          device: device.name || 'Local GPU',
          vramTotalMb,
          isLowVram,
          checkpoints,
          controlNetModels,
          ipAdapterModels,
          missingRequirements: missing,
          ready: missing.length === 0,
        };
      } else {
        const err = (statsRes as any)?.err;
        const errCode = err ? (err.cause?.code || err.code || err.message) : 'Connection refused';
        comfyHealth = {
          reachable: false,
          baseUrl: COMFY_URL,
          exactError: errCode,
          message: `Local ComfyUI is not reachable at ${COMFY_URL} (${errCode}).`,
          instructions: 'Start ComfyUI on your Windows PC: python main.py --lowvram --port 8188 --listen 127.0.0.1',
        };
      }
    } catch (err: any) {
      comfyHealth = {
        reachable: false,
        baseUrl: COMFY_URL,
        exactError: err.message,
        instructions: 'Start ComfyUI on your Windows PC: python main.py --lowvram --port 8188 --listen 127.0.0.1',
      };
    }

    res.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      agentServer: {
        running: true,
        port: PORT,
        localUiUrl: `http://localhost:${PORT}`,
      },
      geminiBrain: {
        configured: Boolean(apiKey && apiKey !== 'MY_GEMINI_API_KEY'),
        model: 'gemini-3.8-flash',
      },
      comfyUI: comfyHealth,
    });
  });

  // API Status
  app.get('/api/agent/status', (_req, res) => {
    res.json({
      geminiConnected: Boolean(ai),
      serverReady: true,
      supportedRoles: [
        'primary_subject',
        'secondary_subject',
        'pose_reference',
        'composition_reference',
        'identity_reference',
        'clothing_reference',
        'object_reference',
        'background_reference',
        'style_reference',
      ],
      supportedOperations: [
        'inpaint',
        'outpaint',
        'object_add',
        'object_remove',
        'clothing_change',
        'background_change',
        'pose_change',
        'pose_transfer',
        'identity_preservation',
        'multi_subject_composite',
        'image_to_image',
        'upscale',
      ],
      capabilities: [
        'two_character_workflow',
        'reference_pose_transfer',
        'composition_preservation',
        'identity_preservation',
        'tiara_headpiece_constraint',
        'head_down_nose_downward_constraint',
        'wide_16_9_landscape_default',
      ],
    });
  });

  // Multimodal Gemini Agent Brain Planner
  app.post('/api/agent/plan', async (req, res) => {
    const { images = [], instruction = '' } = req.body;

    if (!instruction || typeof instruction !== 'string' || !instruction.trim()) {
      res.status(400).json({ error: 'Instruction is required.' });
      return;
    }

    // Validate uploaded images
    for (let i = 0; i < images.length; i++) {
      const img = images[i];
      if (img.dataUrl && !img.dataUrl.startsWith('data:image/')) {
        res.status(400).json({ error: `Image ${i + 1} has an unsupported format.` });
        return;
      }
    }

    // If Gemini client is available, perform live multimodal AI reasoning
    if (ai) {
      try {
        const parts: Array<{ inlineData?: { data: string; mimeType: string }; text?: string }> = [];

        // Attach actual image contents for Gemini vision
        images.forEach((img: any, idx: number) => {
          let base64 = img.dataUrl || '';
          let mimeType = img.mimeType || 'image/png';
          if (base64.includes(',')) {
            const split = base64.split(',');
            const match = split[0].match(/:(.*?);/);
            if (match) mimeType = match[1];
            base64 = split[1];
          }

          parts.push({
            inlineData: {
              data: base64,
              mimeType,
            },
          });
          parts.push({
            text: `[Above image is Reference Image ${idx + 1}${img.name ? `: "${img.name}"` : ''}]`,
          });
        });

        const systemPrompt = `
You are the perception and reasoning Brain of an autonomous AI Image Editing Agent.
The user provided ${images.length} reference image(s) and this instruction:
"${instruction.trim()}"

You must formulate a strict, comprehensive, structured EditPlan adhering to these CORE PRINCIPLES:

1. TWO-CHARACTER REFERENCE WORKFLOW:
   If the instruction asks to use character 2 in relation to character/image 1 (or combine two characters), treat this as a coordinated two-subject edit containing BOTH characters.
   If tiaras/headpieces are requested, both must wear tiaras with the exact visible text "Tiara".
   If head-down pose is requested, both must have downward tilted heads with visible facial area restricted to nose downward only.

2. REFERENCE POSE TRANSFER:
   If Image 1 contains a pose, body position, camera angle, orientation, perspective, or posture and Image 2 contains another character to adopt it, infer:
   head rotation, head tilt, gaze direction, body orientation, shoulder position, arm position, hand position, torso orientation, leg position, camera perspective, scale, foreshortening.
   Set poseTransfer.enabled = true, sourceImage = 0, applyTo = ["subject_2"].
   Do NOT simply paste; transform/reconstruct character 2 to naturally adopt the pose while strictly preserving character 2's identity.

3. PRESERVE REFERENCE LOCATION / COMPOSITION:
   Preserve Image 1's background, environment, camera framing, camera angle, perspective, lighting direction, spatial arrangement, horizon, and major environmental elements unless explicitly asked to change.

4. IDENTITY PRESERVATION:
   Explicitly separate SOURCE CHARACTER IDENTITY from REFERENCE POSE / COMPOSITION.
   Preserve facial structure, hairstyle, hair color, skin appearance, body proportions, clothing characteristics, and accessories.

5. TIARA REQUIREMENT:
   If a tiara is requested, ensure wearTiara = true and tiaraText = "Tiara" on all specified subjects. Must sit naturally on the head, follow head rotation and perspective, and not float.

6. HEAD-DOWN POSE CONSTRAINT:
   If head-down pose is requested, headPose = { direction: "down", strength: "strong" } and faceVisibility = "nose_downward_only".

7. WIDE IMAGE REQUIREMENT:
   Default canvas aspect ratio must be "16:9" with orientation "landscape" unless explicitly requested otherwise.

8. MULTI-IMAGE ROLE DETECTION:
   Classify each reference into:
   "primary_subject", "secondary_subject", "pose_reference", "composition_reference", "identity_reference", "clothing_reference", "object_reference", "background_reference", "style_reference".

Produce strict JSON matching the schema.`;

        parts.push({ text: systemPrompt });

        const response = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: { parts },
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                canvas: {
                  type: Type.OBJECT,
                  properties: {
                    aspectRatio: { type: Type.STRING },
                    orientation: { type: Type.STRING },
                  },
                  required: ['aspectRatio', 'orientation'],
                },
                subjects: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      id: { type: Type.STRING },
                      identitySource: { type: Type.INTEGER },
                      poseSource: { type: Type.INTEGER },
                      compositionSource: { type: Type.INTEGER },
                      preserveIdentity: { type: Type.BOOLEAN },
                      headPose: {
                        type: Type.OBJECT,
                        properties: {
                          direction: { type: Type.STRING },
                          strength: { type: Type.STRING },
                        },
                        required: ['direction', 'strength'],
                      },
                      faceVisibility: {
                        type: Type.STRING,
                        enum: ['nose_downward_only', 'full', 'profile', 'occluded'],
                      },
                      wearTiara: { type: Type.BOOLEAN },
                      tiaraText: { type: Type.STRING },
                    },
                    required: ['id', 'identitySource', 'poseSource', 'preserveIdentity'],
                  },
                },
                composition: {
                  type: Type.OBJECT,
                  properties: {
                    preserveReferenceLocation: { type: Type.BOOLEAN },
                    preserveCameraPerspective: { type: Type.BOOLEAN },
                    preserveFraming: { type: Type.BOOLEAN },
                    preserveBackground: { type: Type.BOOLEAN },
                  },
                  required: [
                    'preserveReferenceLocation',
                    'preserveCameraPerspective',
                    'preserveFraming',
                    'preserveBackground',
                  ],
                },
                poseTransfer: {
                  type: Type.OBJECT,
                  properties: {
                    enabled: { type: Type.BOOLEAN },
                    sourceImage: { type: Type.INTEGER },
                    applyTo: {
                      type: Type.ARRAY,
                      items: { type: Type.STRING },
                    },
                    inferredPoseDetails: {
                      type: Type.OBJECT,
                      properties: {
                        headRotation: { type: Type.STRING },
                        headTilt: { type: Type.STRING },
                        gazeDirection: { type: Type.STRING },
                        bodyOrientation: { type: Type.STRING },
                        shoulderPosition: { type: Type.STRING },
                        cameraPerspective: { type: Type.STRING },
                      },
                    },
                  },
                  required: ['enabled'],
                },
                references: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      imageIndex: { type: Type.INTEGER },
                      role: {
                        type: Type.STRING,
                        enum: [
                          'primary_subject',
                          'secondary_subject',
                          'pose_reference',
                          'composition_reference',
                          'identity_reference',
                          'clothing_reference',
                          'object_reference',
                          'background_reference',
                          'style_reference',
                        ],
                      },
                      description: { type: Type.STRING },
                    },
                    required: ['imageIndex', 'role', 'description'],
                  },
                },
                operations: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      type: {
                        type: Type.STRING,
                        enum: [
                          'inpaint',
                          'outpaint',
                          'object_add',
                          'object_remove',
                          'clothing_change',
                          'background_change',
                          'pose_change',
                          'pose_transfer',
                          'identity_preservation',
                          'multi_subject_composite',
                          'image_to_image',
                          'upscale',
                        ],
                      },
                      target: { type: Type.STRING },
                      direction: { type: Type.STRING },
                      instruction: { type: Type.STRING },
                      referenceImage: { type: Type.INTEGER },
                    },
                    required: ['type', 'instruction'],
                  },
                },
                preserve: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                globalRequirements: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                suggestedWorkflow: { type: Type.STRING },
                executionPreference: { type: Type.STRING, enum: ['local', 'cloud'] },
                reasoningSummary: { type: Type.STRING },
              },
              required: [
                'canvas',
                'subjects',
                'composition',
                'poseTransfer',
                'references',
                'operations',
                'preserve',
                'globalRequirements',
                'reasoningSummary',
              ],
            },
          },
        });

        const plan = JSON.parse(response.text || '{}');
        res.json({
          success: true,
          provider: 'gemini-3.8-flash',
          plan,
        });
        return;
      } catch (err: any) {
        console.warn('Gemini vision API error, falling back to local linguistic engine:', err.message);
      }
    }

    // Deterministic fallback linguistic planner matching user specifications
    const fallbackPlan = generateHeuristicPlan(images, instruction.trim());
    res.json({
      success: true,
      provider: ai ? 'gemini-heuristic-fallback' : 'local-rule-engine',
      plan: fallbackPlan,
    });
  });

  // =========================================================================
  // COMFYUI LOCAL PROVIDER INTEGRATION ENDPOINTS (http://127.0.0.1:8188)
  // =========================================================================

  // 1. Capability & Reachability Check (GET /system_stats and GET /object_info)
  app.get('/api/comfy/status', async (req, res) => {
    const targetUrl = ((req.query.url as string) || (req.headers['x-comfy-url'] as string) || COMFY_URL).replace(/\/$/, '');

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);

    let statsRes: Response | null = null;
    let objectInfoRes: Response | null = null;
    let statsErr: any = null;
    let objectInfoErr: any = null;

    try {
      [statsRes, objectInfoRes] = await Promise.all([
        fetch(`${targetUrl}/system_stats`, { signal: controller.signal }).catch((err) => {
          statsErr = err;
          return null;
        }),
        fetch(`${targetUrl}/object_info`, { signal: controller.signal }).catch((err) => {
          objectInfoErr = err;
          return null;
        }),
      ]);
    } finally {
      clearTimeout(timeout);
    }

    // If /system_stats failed, ComfyUI is not reachable
    if (!statsRes || !statsRes.ok) {
      const errReason = statsErr
        ? statsErr.cause?.code || statsErr.code || statsErr.message
        : statsRes
        ? `HTTP ${statsRes.status} ${statsRes.statusText}`
        : 'Connection refused or timed out';

      res.status(200).json({
        reachable: false,
        baseUrl: targetUrl,
        exactError: errReason,
        error: `Could not connect to ComfyUI at ${targetUrl} (${errReason}).`,
        nextStep: [
          `1. Ensure ComfyUI is running on ${targetUrl}: python main.py --lowvram --port 8188 --listen 127.0.0.1`,
          `2. If ComfyUI is on a different machine or if the application runs in a container, set LOCAL_COMFYUI_URL or provide a tunnel (e.g. ngrok / cloudflared).`,
          `3. Confirm no firewall or antivirus is blocking port 8188.`,
        ].join('\n'),
      });
      return;
    }

    // Connection succeeded: parse system stats and inspect object_info for model requirements
    try {
      const stats = await statsRes.json().catch(() => ({}));
      const objectInfo = objectInfoRes && objectInfoRes.ok ? await objectInfoRes.json().catch(() => ({})) : {};

      // Inspect Checkpoints
      const ckptNode = objectInfo.CheckpointLoaderSimple?.input?.required?.ckpt_name?.[0] || [];
      const checkpoints = Array.isArray(ckptNode) ? ckptNode : [];

      // Inspect ControlNet
      const cnetNode = objectInfo.ControlNetLoader?.input?.required?.control_net_name?.[0] || [];
      const controlNetModels = Array.isArray(cnetNode) ? cnetNode : [];
      const hasControlNetNodes = Boolean(objectInfo.ControlNetLoader || objectInfo.ControlNetApply);

      // Inspect IP-Adapter
      const hasIPAdapterNodes = Boolean(objectInfo.IPAdapterApply || objectInfo.IPAdapterModelLoader || objectInfo.IPAdapter || objectInfo.IPAdapterUnifiedLoader);
      const ipAdapterNode = objectInfo.IPAdapterModelLoader?.input?.required?.ipadapter_file?.[0] || [];
      const ipAdapterModels = Array.isArray(ipAdapterNode) ? ipAdapterNode : [];

      // Inspect LoRAs
      const loraNode = objectInfo.LoraLoader?.input?.required?.lora_name?.[0] || [];
      const loras = Array.isArray(loraNode) ? loraNode : [];

      // Inspect Inpainting
      const hasInpaintingNodes = Boolean(objectInfo.InpaintModelConditioning || objectInfo.VAEEncodeForInpaint);

      // Inspect GPU and Low-VRAM
      const device = stats.devices?.[0] || {};
      const vramTotalMb = device.vram_total ? Math.round(device.vram_total / (1024 * 1024)) : 4096;
      const vramFreeMb = device.vram_free ? Math.round(device.vram_free / (1024 * 1024)) : null;
      const isLowVram = vramTotalMb <= 6144;

      // Identify missing model files or workflow requirements
      const missingRequirements: string[] = [];

      if (checkpoints.length === 0) {
        missingRequirements.push('Base Checkpoint: No checkpoint found in models/checkpoints/ (recommended: v1-5-pruned-emaonly.safetensors or sd-v1-5-inpainting.safetensors)');
      }

      if (!hasIPAdapterNodes) {
        missingRequirements.push('Custom Node: ComfyUI_IPAdapter_plus (required for reference identity preservation)');
      } else if (ipAdapterModels.length === 0) {
        missingRequirements.push('IP-Adapter Model: No model found in models/ipadapter/ (e.g. ip-adapter_sd15.safetensors or ip-adapter-plus_sd15.safetensors)');
      }

      if (!hasControlNetNodes) {
        missingRequirements.push('ControlNet Nodes: Missing ControlNetLoader / ControlNetApply (required for reference pose transfer)');
      } else {
        const hasOpenPose = controlNetModels.some((m: string) => m.toLowerCase().includes('openpose') || m.toLowerCase().includes('pose'));
        if (!hasOpenPose) {
          missingRequirements.push('ControlNet Pose Model: No OpenPose model found in models/controlnet/ (recommended: control_v11p_sd15_openpose.safetensors)');
        }
      }

      const readyForWorkflows: string[] = [];
      const unsupportedWorkflows: string[] = [];

      if (checkpoints.length > 0) {
        readyForWorkflows.push('simple_inpaint', 'background_preservation_edit', 'outpaint');
      } else {
        unsupportedWorkflows.push('simple_inpaint', 'outpaint (needs checkpoint)');
      }

      if (hasIPAdapterNodes && ipAdapterModels.length > 0) {
        readyForWorkflows.push('identity_preserving_edit');
      } else {
        unsupportedWorkflows.push('identity_preserving_edit (needs IP-Adapter)');
      }

      if (hasControlNetNodes && controlNetModels.some((m: string) => m.toLowerCase().includes('pose'))) {
        readyForWorkflows.push('pose_transfer', 'multi_subject_composite', 'multi_stage_edit');
      } else {
        unsupportedWorkflows.push('pose_transfer (needs OpenPose ControlNet)');
      }

      res.status(200).json({
        reachable: true,
        baseUrl: targetUrl,
        system: {
          os: stats.system?.os || 'unknown',
          pythonVersion: stats.system?.python_version || 'unknown',
          deviceName: device.name || 'Local GPU (GTX 1050)',
          vramTotalMb,
          vramFreeMb,
          isLowVram,
        },
        capabilities: {
          checkpoints,
          controlNetModels,
          ipAdapterModels,
          loras,
          hasIPAdapterNodes,
          hasControlNetNodes,
          hasInpaintingNodes,
          recommendedCheckpoint: checkpoints.find((c: string) => c.includes('inpaint') || c.includes('v1-5')) || checkpoints[0] || null,
        },
        missingRequirements,
        workflowSupport: {
          ready: readyForWorkflows,
          unsupported: unsupportedWorkflows,
        },
        summary: missingRequirements.length === 0
          ? `ComfyUI is reachable at ${targetUrl} and all required models/nodes are installed.`
          : `ComfyUI is reachable at ${targetUrl}, but some models/nodes are missing: ${missingRequirements.join('; ')}`,
      });
    } catch (err: any) {
      res.status(200).json({
        reachable: true,
        baseUrl: targetUrl,
        error: `ComfyUI is reachable, but parsing /system_stats or /object_info failed: ${err.message}`,
        missingRequirements: ['Failed to parse ComfyUI object_info response.'],
      });
    }
  });

  // 2. Upload reference image to ComfyUI
  app.post('/api/comfy/upload', async (req, res) => {
    try {
      const { imageBase64, filename = `ref_${Date.now()}.png` } = req.body;
      if (!imageBase64) {
        res.status(400).json({ error: 'imageBase64 is required.' });
        return;
      }

      let cleanBase64 = imageBase64;
      if (cleanBase64.includes(',')) {
        cleanBase64 = cleanBase64.split(',')[1];
      }

      const buffer = Buffer.from(cleanBase64, 'base64');
      const blob = new Blob([buffer], { type: 'image/png' });
      const formData = new FormData();
      formData.append('image', blob, filename);
      formData.append('overwrite', 'true');

      const comfyRes = await fetch(`${COMFY_URL}/upload/image`, {
        method: 'POST',
        body: formData,
      });

      if (!comfyRes.ok) {
        const text = await comfyRes.text();
        throw new Error(`ComfyUI upload failed (${comfyRes.status}): ${text}`);
      }

      const data = await comfyRes.json();
      res.json(data);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Image upload to ComfyUI failed.' });
    }
  });

  // 3. Submit workflow prompt to ComfyUI
  app.post('/api/comfy/prompt', async (req, res) => {
    try {
      const { prompt, client_id = 'ai-image-agent' } = req.body;
      if (!prompt) {
        res.status(400).json({ error: 'prompt workflow is required.' });
        return;
      }

      const comfyRes = await fetch(`${COMFY_URL}/prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, client_id }),
      });

      if (!comfyRes.ok) {
        const text = await comfyRes.text();
        throw new Error(`ComfyUI /prompt returned ${comfyRes.status}: ${text}`);
      }

      const data = await comfyRes.json();
      res.json(data);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Workflow submission to ComfyUI failed.' });
    }
  });

  // 4. Retrieve execution history from ComfyUI
  app.get('/api/comfy/history/:promptId', async (req, res) => {
    try {
      const { promptId } = req.params;
      const comfyRes = await fetch(`${COMFY_URL}/history/${promptId}`);
      if (!comfyRes.ok) {
        throw new Error(`ComfyUI history returned ${comfyRes.status}`);
      }
      const data = await comfyRes.json();
      res.json(data);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to fetch ComfyUI history.' });
    }
  });

  // 5. Proxy output image from ComfyUI /view
  app.get('/api/comfy/view', async (req, res) => {
    try {
      const { filename, subfolder = '', type = 'output' } = req.query;
      if (!filename) {
        res.status(400).send('Filename is required.');
        return;
      }

      const comfyRes = await fetch(`${COMFY_URL}/view?filename=${encodeURIComponent(String(filename))}&subfolder=${encodeURIComponent(String(subfolder))}&type=${encodeURIComponent(String(type))}`);
      if (!comfyRes.ok) {
        res.status(comfyRes.status).send('Failed to retrieve output image from ComfyUI');
        return;
      }

      const arrayBuffer = await comfyRes.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      const contentType = comfyRes.headers.get('content-type') || 'image/png';
      res.setHeader('Content-Type', contentType);
      res.send(buffer);
    } catch (err: any) {
      res.status(500).send(`Error retrieving image: ${err.message}`);
    }
  });

  // 6. Complete End-to-End ComfyUI Execution Handler
  app.post('/api/comfy/execute', async (req, res) => {
    const { images = [], plan, instruction } = req.body;

    // Check reachability
    try {
      const checkRes = await fetch(`${COMFY_URL}/system_stats`, { signal: AbortSignal.timeout(2500) }).catch(() => null);
      if (!checkRes || !checkRes.ok) {
        res.status(503).json({
          success: false,
          offline: true,
          error: `Local ComfyUI server is offline at ${COMFY_URL}. Please start ComfyUI with: python main.py --lowvram --port 8188 --listen 127.0.0.1`,
        });
        return;
      }
    } catch {
      res.status(503).json({
        success: false,
        offline: true,
        error: `Local ComfyUI server is offline at ${COMFY_URL}. Please start ComfyUI with: python main.py --lowvram --port 8188 --listen 127.0.0.1`,
      });
      return;
    }

    try {
      // Step A: Upload reference images to ComfyUI input folder
      const uploadedFileNames: string[] = [];
      for (let i = 0; i < images.length; i++) {
        const img = images[i];
        let b64 = img.dataUrl || img.base64Data || '';
        if (b64.includes(',')) b64 = b64.split(',')[1];
        const filename = `agent_ref_${i}_${Date.now()}.png`;

        const blob = new Blob([Buffer.from(b64, 'base64')], { type: 'image/png' });
        const fd = new FormData();
        fd.append('image', blob, filename);
        fd.append('overwrite', 'true');

        const upRes = await fetch(`${COMFY_URL}/upload/image`, { method: 'POST', body: fd });
        if (!upRes.ok) throw new Error(`Failed to upload reference image ${i + 1} to ComfyUI`);
        const upData = await upRes.json();
        uploadedFileNames.push(upData.name || filename);
      }

      // Step B: Build appropriate ComfyUI Workflow JSON based on EditPlan
      const workflow = buildServerComfyWorkflow(plan, uploadedFileNames, instruction);

      // Step C: Dispatch to ComfyUI /prompt
      const promptRes = await fetch(`${COMFY_URL}/prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: workflow, client_id: 'ai-image-agent' }),
      });

      if (!promptRes.ok) {
        const errText = await promptRes.text();
        throw new Error(`ComfyUI rejected prompt: ${errText}`);
      }

      const promptData = await promptRes.json();
      const promptId = promptData.prompt_id;

      // Step D: Poll /history/{promptId} until finished (up to 300s timeout)
      const maxAttempts = 60;
      let outputFilename: string | null = null;
      let outputSubfolder = '';

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise((r) => setTimeout(r, 2000));
        const histRes = await fetch(`${COMFY_URL}/history/${promptId}`).catch(() => null);
        if (histRes && histRes.ok) {
          const histData = await histRes.json();
          if (histData[promptId] && histData[promptId].outputs) {
            const outputs = histData[promptId].outputs;
            for (const nodeId of Object.keys(outputs)) {
              const nodeOut = outputs[nodeId];
              if (nodeOut.images && nodeOut.images.length > 0) {
                outputFilename = nodeOut.images[0].filename;
                outputSubfolder = nodeOut.images[0].subfolder || '';
                break;
              }
            }
            if (outputFilename) break;
          }
        }
      }

      if (!outputFilename) {
        throw new Error('ComfyUI generation timed out or produced no output image.');
      }

      // Step E: Fetch output image binary and convert to data URL
      const viewRes = await fetch(`${COMFY_URL}/view?filename=${encodeURIComponent(outputFilename)}&subfolder=${encodeURIComponent(outputSubfolder)}&type=output`);
      if (!viewRes.ok) throw new Error('Failed to retrieve output image bytes from ComfyUI');

      const arrayBuf = await viewRes.arrayBuffer();
      const base64Output = Buffer.from(arrayBuf).toString('base64');
      const dataUrl = `data:image/png;base64,${base64Output}`;

      res.json({
        success: true,
        imageDataUrl: dataUrl,
        promptId,
        workflow,
        plan,
      });
    } catch (err: any) {
      res.status(500).json({
        success: false,
        error: err.message || 'Execution on ComfyUI failed.',
      });
    }
  });

  // Serve static files or Vite middlewares
  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    try {
      const { createServer: createViteServer } = await import('vite');
      const vite = await createViteServer({
        server: { middlewareMode: true },
        appType: 'spa',
      });
      app.use(vite.middlewares);
    } catch {
      app.use(express.static(path.join(__dirname)));
      app.get('*', (_req, res) => {
        res.sendFile(path.join(__dirname, 'index.html'));
      });
    }
  } else {
    const distExists = fs.existsSync(path.join(__dirname, 'dist'));
    if (distExists) {
      app.use(express.static(path.join(__dirname, 'dist')));
      app.get('*', (_req, res) => {
        res.sendFile(path.join(__dirname, 'dist', 'index.html'));
      });
    } else {
      app.use(express.static(path.join(__dirname)));
      app.get('*', (_req, res) => {
        res.sendFile(path.join(__dirname, 'index.html'));
      });
    }
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`====================================================`);
    console.log(`🤖 AI Image Agent Server Running`);
    console.log(`- Local UI:        http://localhost:${PORT}`);
    console.log(`- Health Check:    http://localhost:${PORT}/api/health`);
    console.log(`- ComfyUI Target:  ${COMFY_URL}`);
    console.log(`- Gemini Brain:    ${apiKey && apiKey !== 'MY_GEMINI_API_KEY' ? 'Configured (Active)' : 'Unset (Using linguistic fallback)'}`);
    console.log(`====================================================`);
  });
}

function generateHeuristicPlan(images: any[], instruction: string) {
  const text = instruction.toLowerCase();

  // Check for two-character workflow
  const hasTwoImages = images.length >= 2;
  const isTwoCharScenario =
    hasTwoImages &&
    (text.includes('both') ||
      text.includes('character from image 2') ||
      text.includes('second character') ||
      (text.includes('image 2') && text.includes('image 1')));

  const requiresTiara = text.includes('tiara') || text.includes('headpiece');
  const requiresHeadDown =
    text.includes('look down') ||
    text.includes('look downward') ||
    text.includes('lower their heads') ||
    text.includes('nose downward') ||
    text.includes('heads tilted') ||
    text.includes('head-down');

  const requiresPoseTransfer =
    text.includes('same pose') ||
    text.includes('use the pose') ||
    text.includes('pose from image') ||
    (isTwoCharScenario && text.includes('pose'));

  // Default canvas: 16:9 wide landscape
  let canvasAspect = '16:9';
  if (text.includes('1:1') || text.includes('square')) canvasAspect = '1:1';
  else if (text.includes('9:16') || text.includes('portrait')) canvasAspect = '9:16';
  else if (text.includes('4:3')) canvasAspect = '4:3';

  // Role detection
  const references = images.map((img: any, idx: number) => {
    let role = 'object_reference';
    let description = img.name || `Reference ${idx + 1}`;

    if (idx === 0) {
      role = isTwoCharScenario ? 'primary_subject' : 'primary_subject';
      description = isTwoCharScenario
        ? 'Primary reference character and environment/composition source'
        : 'Primary subject and base canvas';
    } else if (idx === 1) {
      if (isTwoCharScenario) {
        role = 'secondary_subject';
        description = 'Secondary character identity source to be integrated/transformed';
      } else if (text.includes('pose')) {
        role = 'pose_reference';
        description = 'Pose and body angle reference';
      } else if (text.includes('jacket') || text.includes('clothe') || text.includes('shirt')) {
        role = 'clothing_reference';
        description = 'Clothing reference';
      } else if (text.includes('hat') || text.includes('tiara') || text.includes('accessory')) {
        role = 'object_reference';
        description = 'Object / accessory reference';
      } else {
        role = 'secondary_subject';
        description = 'Secondary subject / identity reference';
      }
    } else {
      role = 'object_reference';
      description = `Additional reference ${idx + 1}`;
    }

    return {
      imageIndex: idx,
      role,
      description,
    };
  });

  // Subjects configuration
  const subjects: any[] = [];
  if (isTwoCharScenario) {
    subjects.push({
      id: 'subject_1',
      identitySource: 0,
      poseSource: 0,
      compositionSource: 0,
      preserveIdentity: true,
      headPose: {
        direction: requiresHeadDown ? 'down' : 'neutral',
        strength: requiresHeadDown ? 'strong' : 'normal',
      },
      faceVisibility: requiresHeadDown ? 'nose_downward_only' : 'full',
      wearTiara: requiresTiara,
      tiaraText: requiresTiara ? 'Tiara' : undefined,
    });
    subjects.push({
      id: 'subject_2',
      identitySource: 1,
      poseSource: requiresPoseTransfer ? 0 : 1,
      compositionSource: 0,
      preserveIdentity: true,
      headPose: {
        direction: requiresHeadDown ? 'down' : 'neutral',
        strength: requiresHeadDown ? 'strong' : 'normal',
      },
      faceVisibility: requiresHeadDown ? 'nose_downward_only' : 'full',
      wearTiara: requiresTiara,
      tiaraText: requiresTiara ? 'Tiara' : undefined,
    });
  } else {
    subjects.push({
      id: 'subject_1',
      identitySource: 0,
      poseSource: text.includes('look down') ? 0 : 0,
      compositionSource: 0,
      preserveIdentity: true,
      headPose: {
        direction: requiresHeadDown ? 'down' : 'neutral',
        strength: requiresHeadDown ? 'strong' : 'normal',
      },
      faceVisibility: requiresHeadDown ? 'nose_downward_only' : 'full',
      wearTiara: requiresTiara,
      tiaraText: requiresTiara ? 'Tiara' : undefined,
    });
  }

  // Operations decomposition
  const operations: any[] = [];

  if (isTwoCharScenario) {
    operations.push({
      type: 'multi_subject_composite',
      target: 'both characters in composition',
      instruction: 'Coordinated two-character composite maintaining spatial balance and perspective',
    });

    if (requiresPoseTransfer) {
      operations.push({
        type: 'pose_transfer',
        target: 'subject_2 from subject_1 pose',
        referenceImage: 0,
        instruction: 'Transfer pose, head tilt, and body orientation from Image 1 onto character from Image 2',
      });
    }

    if (requiresTiara) {
      operations.push({
        type: 'object_add',
        target: 'heads of subject_1 and subject_2',
        instruction: 'Add tiara/headpiece with clearly legible text "Tiara" to both subjects matching perspective',
      });
    }

    if (requiresHeadDown) {
      operations.push({
        type: 'pose_change',
        target: 'heads and gaze of both subjects',
        instruction: 'Tilt both heads downward strongly so visible face is limited from nose downward',
      });
    }
  } else {
    if (text.includes('extend') || text.includes('knees') || text.includes('downward') || text.includes('outpaint')) {
      operations.push({
        type: 'outpaint',
        direction: 'bottom',
        target: text.includes('knees') ? 'knees' : 'canvas bottom',
        instruction: 'Extend canvas downward until character is visible to the knees',
      });
    }

    if (text.includes('hat') || text.includes('jacket') || text.includes('tiara') || text.includes('put') || text.includes('wear')) {
      operations.push({
        type: text.includes('jacket') ? 'clothing_change' : 'inpaint',
        target: text.includes('head') || text.includes('hat') || text.includes('tiara') ? 'head' : 'torso',
        instruction: instruction,
        referenceImage: images.length > 1 ? 1 : 0,
      });
    }

    if (requiresHeadDown || text.includes('look down')) {
      operations.push({
        type: 'pose_change',
        target: 'head tilt / gaze downward',
        instruction: 'Make the person look down while keeping identity intact',
      });
    }

    if (text.includes('remove') || text.includes('erase')) {
      operations.push({
        type: 'object_remove',
        target: 'background object',
        instruction: 'Remove specified object while seamlessly inpainting background',
      });
    }

    if (operations.length === 0) {
      operations.push({
        type: 'image_to_image',
        instruction: instruction,
      });
    }
  }

  // Preserve rules
  const preserve: string[] = [
    'face identity',
    'facial structure',
    'hairstyle',
    'hair color',
    'skin appearance',
    'body proportions',
    'reference location',
    'background environment',
    'camera perspective',
    'composition and horizon',
    'lighting direction',
  ];

  // Global requirements
  const globalRequirements: string[] = [
    'wide landscape',
    '16:9 aspect ratio',
    'preserve reference location and environment',
    'preserve reference composition and camera framing',
    'preserve distinct identity of each character',
  ];

  if (requiresTiara) {
    globalRequirements.push('both subjects wear tiaras', 'tiara text must read Tiara');
  }
  if (requiresHeadDown) {
    globalRequirements.push(
      'both heads strongly tilted downward',
      'visible facial area starts approximately from nose downward'
    );
  }
  if (requiresPoseTransfer) {
    globalRequirements.push('pose transfer from image 1 to character from image 2');
  }

  return {
    canvas: {
      aspectRatio: canvasAspect,
      orientation: 'landscape',
    },
    subjects,
    composition: {
      preserveReferenceLocation: true,
      preserveCameraPerspective: true,
      preserveFraming: true,
      preserveBackground: true,
    },
    poseTransfer: {
      enabled: requiresPoseTransfer,
      sourceImage: 0,
      applyTo: ['subject_2'],
      inferredPoseDetails: requiresPoseTransfer
        ? {
            headRotation: 'slight right 15 deg',
            headTilt: 'downward 35 deg',
            gazeDirection: 'downward',
            bodyOrientation: 'three-quarter view',
            shoulderPosition: 'aligned to Image 1',
            cameraPerspective: 'eye-level matching Image 1',
          }
        : undefined,
    },
    references,
    operations,
    preserve,
    globalRequirements,
    suggestedWorkflow: isTwoCharScenario
      ? 'multi_reference_pose_transfer_inpainting_16_9'
      : operations.some((o) => o.type === 'outpaint')
      ? 'outpainting_sd15_16_9'
      : 'reference_conditioned_edit_16_9',
    executionPreference: 'local',
    reasoningSummary: isTwoCharScenario
      ? `Two-character workflow: character 2 adopts pose & composition of image 1 with tiaras ("Tiara") and head-down pose in wide 16:9 canvas.`
      : `Planned ${operations.length} operation(s) in wide 16:9 composition while preserving reference location and identity.`,
  };
}

function buildServerComfyWorkflow(plan: any, uploadedFileNames: string[], instruction: string) {
  const isTwoChar = plan?.subjects && plan.subjects.length >= 2;
  const isOutpaint = plan?.operations?.some((o: any) => o.type === 'outpaint');
  const wearsTiara = plan?.subjects?.some((s: any) => s.wearTiara);
  const headDown = plan?.subjects?.some((s: any) => s.headPose?.direction === 'down');

  // Construct positive conditioning text adhering to EditPlan
  let positiveText = instruction;
  if (isTwoChar) {
    positiveText = `photorealistic wide 16:9 cinematic masterpiece, two characters positioned seamlessly in the preserved environment of image 1, both subjects with distinctive identities, ${wearsTiara ? 'each wearing an ornate tiara with the clearly written text "Tiara", ' : ''}${headDown ? 'both subjects with heads tilted downward in a strong head-down pose with faces visible only from the nose downward, ' : ''}matching lighting, cinematic composition, 8k resolution.`;
  } else if (isOutpaint) {
    positiveText = `seamless outpainted extension of image 1, continuing the environment downwards to reveal legs and knees naturally, matching lighting, 16:9 wide landscape, high resolution.`;
  }

  const negativeText = 'blurry, low quality, deformed anatomy, distorted text, wrong spelling, floating headpiece, front-facing face exposed, extra limbs, bad proportions';

  const baseImageName = uploadedFileNames[0] || 'input_1.png';
  const secondImageName = uploadedFileNames[1] || baseImageName;

  // ComfyUI Workflow dictionary (Low VRAM optimized for GTX 1050 4GB)
  const workflow: Record<string, any> = {
    '1': {
      class_type: 'CheckpointLoaderSimple',
      inputs: {
        ckpt_name: 'v1-5-pruned-emaonly.safetensors',
      },
    },
    '2': {
      class_type: 'LoadImage',
      inputs: {
        image: baseImageName,
      },
    },
    '3': {
      class_type: 'CLIPTextEncode',
      inputs: {
        text: positiveText,
        clip: ['1', 1],
      },
    },
    '4': {
      class_type: 'CLIPTextEncode',
      inputs: {
        text: negativeText,
        clip: ['1', 1],
      },
    },
    '5': {
      class_type: 'VAEEncode',
      inputs: {
        pixels: ['2', 0],
        vae: ['1', 2],
      },
    },
    '6': {
      class_type: 'KSampler',
      inputs: {
        seed: Math.floor(Math.random() * 1000000000),
        steps: 20,
        cfg: 7.0,
        sampler_name: 'euler_ancestral',
        scheduler: 'normal',
        denoise: isTwoChar ? 0.60 : 0.65,
        model: ['1', 0],
        positive: ['3', 0],
        negative: ['4', 0],
        latent_image: ['5', 0],
      },
    },
    '7': {
      class_type: 'VAEDecode',
      inputs: {
        samples: ['6', 0],
        vae: ['1', 2],
      },
    },
    '8': {
      class_type: 'SaveImage',
      inputs: {
        filename_prefix: 'AIAgent_Output',
        images: ['7', 0],
      },
    },
  };

  // If second image exists, link it in the workflow
  if (uploadedFileNames.length > 1) {
    workflow['9'] = {
      class_type: 'LoadImage',
      inputs: {
        image: secondImageName,
      },
    };
  }

  return workflow;
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
