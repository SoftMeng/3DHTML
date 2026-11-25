/* 几何内核：最小三维向量与四元数。
   零依赖（无 DOM / 无 three.js / 无全局状态）⇒ 可被 Node 直接 import 做单测。
   API 与 three.js 对齐（x/y/z、dot/cross/normalize、setFromUnitVectors），
   便于在边界层与 THREE.Vector3 / THREE.Quaternion 互换。 */

export class Vec3 {
  constructor(x = 0, y = 0, z = 0){ this.x = x; this.y = y; this.z = z; }
  clone(){ return new Vec3(this.x, this.y, this.z); }
  set(x, y, z){ this.x = x; this.y = y; this.z = z; return this; }
  copy(v){ this.x = v.x; this.y = v.y; this.z = v.z; return this; }
  add(v){ return new Vec3(this.x + v.x, this.y + v.y, this.z + v.z); }
  sub(v){ return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z); }
  addScaledVector(v, s){ return new Vec3(this.x + v.x * s, this.y + v.y * s, this.z + v.z * s); }
  scale(s){ return new Vec3(this.x * s, this.y * s, this.z * s); }
  dot(v){ return this.x * v.x + this.y * v.y + this.z * v.z; }
  cross(v){ return new Vec3(this.y * v.z - this.z * v.y, this.z * v.x - this.x * v.z, this.x * v.y - this.y * v.x); }
  lengthSq(){ return this.x * this.x + this.y * this.y + this.z * this.z; }
  length(){ return Math.sqrt(this.lengthSq()); }
  distanceTo(v){ return this.sub(v).length(); }
  normalize(){
    const l = this.length();
    return l > 1e-12 ? this.scale(1 / l) : new Vec3(0, 0, 0);
  }  lerp(v, t){ return new Vec3(this.x + (v.x - this.x) * t, this.y + (v.y - this.y) * t, this.z + (v.z - this.z) * t); }
  applyQuat(q){
    const {x, y, z} = this, qx = q.x, qy = q.y, qz = q.z, qw = q.w;
    const ix = qw * x + qy * z - qz * y;
    const iy = qw * y + qz * x - qx * z;
    const iz = qw * z + qx * y - qy * x;
    const iw = -qx * x - qy * y - qz * z;
    return new Vec3(ix * qw + iw * -qx + iy * -qz - iz * -qy,
                    iy * qw + iw * -qy + iz * -qx - ix * -qz,
                    iz * qw + iw * -qz + ix * -qy - iy * -qx);
  }
}

export class Quat {
  constructor(x = 0, y = 0, z = 0, w = 1){ this.x = x; this.y = y; this.z = z; this.w = w; }
  clone(){ return new Quat(this.x, this.y, this.z, this.w); }
  normalize(){
    const l = Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
    if(l < 1e-12) return new Quat(0, 0, 0, 1);
    this.x /= l; this.y /= l; this.z /= l; this.w /= l;
    return this;
  }
  /* 与 three.js Quaternion.setFromUnitVectors 同式：from→to 的最小旋转 */
  static fromUnitVectors(from, to){
    let r = from.dot(to) + 1;
    if(r < Number.EPSILON){
      r = 0;
      if(Math.abs(from.x) > Math.abs(from.z)) return new Quat(-from.y, from.x, 0, r).normalize();
      return new Quat(0, -from.z, from.y, r).normalize();
    }
    return new Quat(from.y * to.z - from.z * to.y,
                    from.z * to.x - from.x * to.z,
                    from.x * to.y - from.y * to.x,
                    r).normalize();
  }
  static axisAngle(axis, angle){
    const h = angle / 2, s = Math.sin(h);
    return new Quat(axis.x * s, axis.y * s, axis.z * s, Math.cos(h));
  }}
