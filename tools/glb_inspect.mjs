#!/usr/bin/env node
/** GLB 结构速查：节点/骨名/蒙皮/动画/贴图，用于评估模型能否直接驱动。 */
import fs from 'fs';

const buf = fs.readFileSync(process.argv[2]);
let off = 12, json = null;
while (off < buf.length) {
  const len = buf.readUInt32LE(off), type = buf.readUInt32LE(off + 4), start = off + 8;
  if (type === 0x4e4f534a) json = JSON.parse(buf.slice(start, start + len).toString('utf8'));
  off = start + len;
}
const nodes = json.nodes || [];
const skins = json.skins || [];
const meshes = json.meshes || [];
const anims = json.animations || [];
const mats = (json.materials || []);
const imgs = json.images || [];
const texs = json.textures || [];
let tris = 0;
for (const m of meshes) {
  for (const p of m.primitives) {
    const idx = p.indices != null ? json.accessors[p.indices].count : (json.accessors[p.attributes.POSITION]?.count || 0);
    tris += idx / 3;
  }
}
const boneNames = nodes.filter(n => n.name && !n.mesh && n.children && n.children.length && !n.skin).map(n => n.name);
const skinJoints = skins.flatMap(s => (s.joints || []).map(i => nodes[i]?.name).filter(Boolean));
const allBones = [...new Set([...boneNames, ...skinJoints])];
console.log(JSON.stringify({
  file: process.argv[2],
  sizeMB: +(buf.length / 1048576).toFixed(2),
  nodes: nodes.length, meshes: meshes.length, skins: skins.length,
  triangles: tris, materials: mats.length, textures: texs.length, images: imgs.length,
  animations: anims.map(a => a.name || '(无名)'),
  bones: allBones.length,
  boneNames: allBones.slice(0, 70),
  hasMixamoNaming: allBones.filter(n => /mixamorig/i.test(n)).length,
  sampleMaterial: mats[0] ? { name: mats[0].name, pbr: !!mats[0].pbrMetallicRoughness, baseColorTex: !!mats[0].pbrMetallicRoughness?.baseColorTexture } : null,
  extensionsUsed: json.extensionsUsed || [],
}, null, 1));
