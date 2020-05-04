#pragma once

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <limits>
#include <memory>
#include <vector>

using atlas::core::areEqual;

using Colour = atlas::math::Vector;

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image);

// Declarations
class BRDF;
class Camera;
class Material;
class Light;
class Shape;
class Sampler;
class Tracer;

struct World
{
	int max_depth;
	std::size_t width, height;
	Colour background;
	std::shared_ptr<Sampler> sampler;
	std::vector<std::shared_ptr<Shape>> scene;
	std::vector<Colour> image;
	std::vector<std::shared_ptr<Light>> lights;
	std::shared_ptr<Light> ambient;
	std::shared_ptr<Camera> camera;
	std::shared_ptr<Tracer> tracer_ptr;
};

struct ShadeRec
{
	Colour color;
	float t;
	int depth;
	atlas::math::Normal normal;
	atlas::math::Ray<atlas::math::Vector> ray;
	atlas::math::Point hit_point;
	std::shared_ptr<Material> material;
	std::shared_ptr<World> world;
};

// Abstract classes defining the interfaces for concrete entities
class Camera
{
public:
	Camera();

	virtual ~Camera() = default;

	virtual void renderScene(std::shared_ptr<World> world) const = 0;

	void setEye(atlas::math::Point const& eye);

	void setLookAt(atlas::math::Point const& lookAt);

	void setUpVector(atlas::math::Vector const& up);

	void computeUVW();

protected:
	atlas::math::Point mEye;
	atlas::math::Point mLookAt;
	atlas::math::Point mUp;
	atlas::math::Vector mU, mV, mW;
};

class Sampler
{
public:
	Sampler(int numSamples, int numSets);

	virtual ~Sampler() = default;

	int getNumSamples() const;

	void setupShuffledIndeces();

	virtual void generateSamples() = 0;

	Sampler&
		operator= (const Sampler& rhs);

	void map_samples_to_hemisphere(const float e);

	void map_samples_to_disk();

	atlas::math::Point sample_unit_hemisphere();

	atlas::math::Point sample_unit_disk();

	atlas::math::Point sampleUnitSquare();

protected:
	std::vector<atlas::math::Point> mSamples;
	std::vector<atlas::math::Point> mHemisphereSample;
	std::vector<atlas::math::Point> disk_samples;
	std::vector<int> mShuffledIndeces;


	int mNumSamples;
	int mNumSets;
	unsigned long mCount;
	unsigned long mCountHemisphere;
	unsigned long mCountDisk;
	int mJump;
};

class Shape
{
public:
	Shape();
	virtual ~Shape() = default;

	// if t computed is less than the t in sr, it and the color should be
	// updated in sr
	virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const = 0;

	virtual bool shadow_hit(atlas::math::Ray<atlas::math::Vector> const& ray, float& t) const = 0;

	virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const = 0;

	void setColour(Colour const& col);

	Colour getColour() const;

	void setMaterial(std::shared_ptr<Material> const& material);

	std::shared_ptr<Material> getMaterial() const;

protected:
	Colour mColour;
	std::shared_ptr<Material> mMaterial;
};

class BRDF
{
public:
	virtual ~BRDF() = default;

	virtual Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const = 0;
	virtual Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const = 0;
protected:
	std::shared_ptr<Sampler> sampler_ptr;
};

class Material
{
public:
	virtual ~Material() = default;

	virtual Colour shade(ShadeRec& sr) = 0;
};

class Light
{
public:
	virtual atlas::math::Vector getDirection(ShadeRec& sr) = 0;

	virtual Colour L(ShadeRec& sr);

	virtual bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, const ShadeRec& sr) const = 0;

	virtual bool cast_shadows() = 0;

	void scaleRadiance(float b);

	void setColour(Colour const& c);

protected:
	Colour mColour;
	float mRadiance;
};

// Concrete classes which we can construct and use in our ray tracer
class Triangle : public Shape
{
public:
	Triangle();

	Triangle(const atlas::math::Point a, atlas::math::Point b, atlas::math::Point c);

	bool shadow_hit(atlas::math::Ray<atlas::math::Vector> const& ray, float& t) const;

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;
private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;
	atlas::math::Point v0, v1, v2;
	atlas::math::Vector normal;
	atlas::math::Vector v0v1;
	atlas::math::Vector v0v2;
};
class Box : public Shape
{
public:
	Box(atlas::math::Point closeCornor, atlas::math::Point farCornor);

	bool shadow_hit(atlas::math::Ray<atlas::math::Vector> const& ray, float& t) const;

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;
private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point closeCornor;
	atlas::math::Point farCornor;
};

class BBox {
public:
	double x0, x1, y0, y1, z0, z1;

	BBox(void);

	BBox(const double x0, const double x1,
		const double y0, const double y1,
		const double z0, const double z1);

	BBox(const atlas::math::Point p0, const atlas::math::Point p1);

	BBox(const BBox& bbox);

	BBox&
		operator= (const BBox& rhs);

	~BBox(void);

	bool
		hit(atlas::math::Ray<atlas::math::Vector> const& ray) const;

	bool
		inside(const atlas::math::Point& p) const;
};

class Rectangle : public Shape {
public:

	Rectangle(const atlas::math::Point& _p0, const atlas::math::Vector& _a, const atlas::math::Vector& _b);

	Rectangle(const atlas::math::Point& _p0, const atlas::math::Vector& _a, const atlas::math::Vector& _b, const atlas::math::Normal& n);

	BBox
		get_bounding_box(void);

	virtual
		bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
			ShadeRec& sr) const;
private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;


	// the following functions are used when the rectangle is a light source

	virtual void
		set_sampler(std::shared_ptr<Sampler> sampler);

	virtual atlas::math::Point
		sample(void);

	virtual atlas::math::Normal
		get_normal(const atlas::math::Point& p);

	virtual float
		pdf(ShadeRec& sr);

private:

	atlas::math::Point 		p0;   			// corner vertex 
	atlas::math::Vector		a;				// side
	atlas::math::Vector		b;				// side
	double			a_len_squared;	// square of the length of side a
	double			b_len_squared;	// square of the length of side b
	atlas::math::Normal			normal;

	float			area;			// for rectangular lights
	float			inv_area;		// for rectangular lights
	std::shared_ptr<Sampler> sampler_ptr;	// for rectangular lights 	

	static const double kEpsilon;
};

class Sphere : public Shape
{
public:
	Sphere(atlas::math::Point center, float radius);

	bool shadow_hit(atlas::math::Ray<atlas::math::Vector> const& ray, float& t) const;

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point mCentre;
	float mRadius;
	float mRadiusSqr;
};

class Plane : public Shape
{
public:
	Plane(const atlas::math::Point point, const atlas::math::Vector normal);

	bool shadow_hit(atlas::math::Ray<atlas::math::Vector> const& ray, float& t) const;

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;
	atlas::math::Vector mnormal;
	atlas::math::Point mpoint;
};

class Pinhole : public Camera
{
public:
	Pinhole();

	void setDistance(float distance);
	void setZoom(float zoom);

	atlas::math::Vector rayDirection(atlas::math::Point const& p) const;
	void renderScene(std::shared_ptr<World> world) const;

private:
	float mDistance;
	float mZoom;
};


class Jittered : public Sampler
{
public:
	Jittered(int numSamples, int numSets);

	void generateSamples();
	Jittered&
		operator= (const Jittered& rhs);
};

class MultiJittered : public Sampler {
public:

	MultiJittered&
		operator= (const MultiJittered& rhs);

	MultiJittered(const int num_samples, const int m);

	void generateSamples();
};

class Lambertian : public BRDF
{
public:
	Lambertian();
	Lambertian(Colour diffuseColor, float diffuseReflection);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	void setDiffuseReflection(float kd);

	void setDiffuseColour(Colour const& colour);

private:
	Colour mDiffuseColour;
	float mDiffuseReflection;
};

class GlossySpecular : public BRDF {
public:

	GlossySpecular(void);
	GlossySpecular::GlossySpecular(Colour specularColour, float diffuseReflection,
		float specularReflection, float exponent);
	~GlossySpecular(void);

	virtual GlossySpecular*
		clone(void) const;

	virtual Colour
		fn(const ShadeRec& sr, const atlas::math::Vector& wo, const atlas::math::Vector& wi) const;

	virtual Colour
		sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wi, float& pdf) const;

	virtual Colour
		rho(const ShadeRec& sr, const atlas::math::Vector& wo) const;

	void setDiffuseReflection(float kd);
	void setSpecularReflection(float ks);
	void setSpecularColour(Colour const& colour);
	void setKr(float pkr);
	void setExponent(float ep);



	void
		set_sampler(std::shared_ptr<Sampler> sp);   			// any type of sampling

	void
		set_sampler(const int num_samples, const int num_sets);   			// any type of sampling



	Colour mSpecularColour;
	float mDiffuseReflection;
	float mSpecularReflection;
	float mExponent;
	float kr;
	std::shared_ptr<Sampler> mSampler;
};


class PerfectSpecular : public BRDF
{
public:

	PerfectSpecular(void);

	~PerfectSpecular(void);

	virtual PerfectSpecular*
		clone(void) const;

	void
		set_kr(const float k);

	void
		set_cr(const Colour& c);

	void
		set_cr(const float r, const float g, const float b);

	void
		set_cr(const float c);

	virtual Colour
		fn(const ShadeRec& sr, const atlas::math::Vector& wo, const atlas::math::Vector& wi) const;

	virtual void
		set_sampler(std::shared_ptr<Sampler> sampler);
	virtual Colour
		sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wi) const;

	virtual Colour
		sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wi, float& pdf) const;

	virtual Colour
		rho(const ShadeRec& sr, const atlas::math::Vector& wo) const;

private:

	float		kr;			// reflection coefficient
	Colour 	cr;			// the reflection colour
	std::shared_ptr<Sampler> mSampler;
};


inline void
PerfectSpecular::set_kr(const float k) {
	kr = k;
}


inline void
PerfectSpecular::set_cr(const Colour& c) {
	cr = c;
}


inline void
PerfectSpecular::set_cr(const float r, const float g, const float b) {
	cr.r = r; cr.g = g; cr.b = b;
}


inline void
PerfectSpecular::set_cr(const float c) {
	cr.r = c; cr.g = c; cr.b = c;
}
class Matte : public Material
{
public:
	Matte();
	Matte(float kd, float ka, Colour color);

	void setDiffuseReflection(float k);

	void setAmbientReflection(float k);

	void setDiffuseColour(Colour colour);

	Colour shade(ShadeRec& sr) override;

private:
	std::shared_ptr<Lambertian> mDiffuseBRDF;
	std::shared_ptr<Lambertian> mAmbientBRDF;
};

class Phong : public Material {
public:

	// contructors, etc ...
	Phong(void);
	Phong(float kd, float ka, Colour color);
	virtual Colour
		shade(ShadeRec& s);

	void Phong::setDiffuseReflection(float k);

	void Phong::setAmbientReflection(float k);

	void Phong::setDiffuseColour(Colour colour);


	inline void
		Phong::set_ks(const float ks) {
		specular_brdf->setSpecularReflection(ks);
	};

	inline void
		set_exp(const float exp) {
		specular_brdf->setExponent(exp);
	}
	inline void
		set_cs(const Colour& c) {
		specular_brdf->setSpecularColour(c);
	}


protected:
	std::shared_ptr<Lambertian> ambient_brdf;
	std::shared_ptr<Lambertian> diffuse_brdf;
	GlossySpecular* specular_brdf;

};

class Directional : public Light
{
public:
	Directional();
	Directional(atlas::math::Vector const& d);

	void setDirection(atlas::math::Vector const& d);

	bool cast_shadows() override;

	bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, const ShadeRec& sr) const override;

	atlas::math::Vector getDirection(ShadeRec& sr) override;

private:
	atlas::math::Vector mDirection;
};

class AmbientOccluder :public Light
{
public:
	AmbientOccluder();

	AmbientOccluder(float min);

	void set_sampler(std::shared_ptr<Sampler> s_ptr);

	atlas::math::Vector getDirection(ShadeRec& sr) override;

	bool cast_shadows() override;

	Colour AmbientOccluder::L(ShadeRec& sr) override;

	bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, const ShadeRec& sr) const override;

private:
	atlas::math::Vector u, v, w;
	float min_amount;
	int count;
};
class Ambient : public Light
{
public:
	Ambient();

	atlas::math::Vector getDirection(ShadeRec& sr) override;

	bool cast_shadows() override;

	bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, const ShadeRec& sr) const override;

private:
	atlas::math::Vector mDirection;
};

class PointLight : public Light
{
public:

	PointLight();

	PointLight(const atlas::math::Vector lo);

	void setLocation(const atlas::math::Vector lo);

	bool cast_shadows() override;

	bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, const ShadeRec& sr) const override;

	atlas::math::Vector getDirection(ShadeRec& sr) override;

private:
	atlas::math::Vector location;
};

class BTDF {
public:
	BTDF(void);

	BTDF(const BTDF& btdf);

	virtual BTDF*
		clone(void) = 0;

	BTDF&
		operator= (const BTDF& rhs);

	virtual
		~BTDF(void);

	virtual Colour
		f(const ShadeRec& sr, const atlas::math::Vector& wo, const atlas::math::Vector& wi) const = 0;

	virtual Colour
		sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wt) const = 0;

	virtual Colour
		rho(const ShadeRec& sr, const atlas::math::Vector& wo) const = 0;
};

class PerfectTransmitter : public BTDF {
public:

	PerfectTransmitter(void);

	PerfectTransmitter(float ior, float kt);

	PerfectTransmitter(const PerfectTransmitter& pt);

	virtual PerfectTransmitter*
		clone(void);

	~PerfectTransmitter(void);

	PerfectTransmitter&
		operator= (const PerfectTransmitter& rhs);

	void
		set_kt(const float k);

	void
		set_colour(Colour c);

	void
		set_ior(const float eta);

	bool
		tir(const ShadeRec& sr) const;

	virtual void
		set_sampler(std::shared_ptr<Sampler> sampler);

	virtual Colour
		f(const ShadeRec& sr, const atlas::math::Vector& wo, const atlas::math::Vector& wi) const;

	virtual Colour
		sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wt) const;

	virtual Colour
		rho(const ShadeRec& sr, const atlas::math::Vector& wo) const;

private:
	Colour mColour;
	float	kt;			// transmission coefficient
	float	ior;		// index of refraction
	std::shared_ptr<Sampler> mSampler;
};

class Tracer {
public:

	Tracer(void);

	Tracer(std::shared_ptr<World> _world_ptr);

	virtual
		~Tracer(void);

	virtual Colour			// only overridden in the tracers SingleSphere and MultipleObjects
		trace_ray(atlas::math::Ray<atlas::math::Vector> const& ray) const;

	virtual Colour
		trace_ray(atlas::math::Ray<atlas::math::Vector> const& ray, const int depth) const;

protected:

	std::shared_ptr<World> world_ptr;
};

class Whitted : public Tracer {
public:

	Whitted(void);

	Whitted(std::shared_ptr<World> _worldPtr);

	virtual
		~Whitted(void);

	virtual Colour
		trace_ray(atlas::math::Ray<atlas::math::Vector> const& ray, const int depth) const;
};

class Regular : public Sampler
{
public:
	Regular(int numSamples, int numSets);

	void generateSamples();
};

class Reflective : public Phong {
public:

	Reflective(void);

	Reflective(float kd, float ka, Colour color);

	Reflective(const Reflective& rm);

	// ---------------------------------------------------------------- set_kr

	inline void
		Reflective::set_kr(const float k) {
		reflective_brdf->set_kr(k);
	}


	// ---------------------------------------------------------------- set_cr

	inline void
		Reflective::set_cr(const Colour& c) {
		reflective_brdf->set_cr(c);

	}


	// ---------------------------------------------------------------- set_cr

	inline void
		Reflective::set_cr(const float r, const float g, const float b) {
		reflective_brdf->set_cr(r, g, b);
	}


	// ---------------------------------------------------------------- set_cr

	inline void
		Reflective::set_cr(const float c) {
		reflective_brdf->set_cr(c);
	}

	virtual Colour
		shade(ShadeRec& s);

private:

	PerfectSpecular* reflective_brdf;
};

class GlossyReflector : public Phong {
public:

	GlossyReflector(void);

	GlossyReflector(float kd, float ka, Colour color);

	void set_samples(std::shared_ptr<Sampler> sampler);

	void set_samples(const int num_samples, const int num_sets);

	inline void
		GlossyReflector::set_cr(const Colour& c) {
		glossy_specular_brdf->setSpecularColour(c);

	}

	void
		set_kr(const float k);

	void set_cs(Colour c);

	void
		set_exponent(const float exp);

	virtual Colour
		shade(ShadeRec& s);

	/*virtual Colour
		area_light_shade(ShadeRec& sr);*/

private:
	GlossySpecular* glossy_specular_brdf;

};

inline void GlossyReflector::set_cs(Colour c)
{
	glossy_specular_brdf->mSpecularColour = c;
}

inline void GlossyReflector::set_samples(std::shared_ptr<Sampler> sampler)
{
	glossy_specular_brdf->set_sampler(sampler);
}

inline void GlossyReflector::set_samples(const int num_samples, const int num_sets)
{
	glossy_specular_brdf->set_sampler(num_samples, num_sets);
}

inline void
GlossyReflector::set_kr(const float k) {
	glossy_specular_brdf->mSpecularReflection = k;
}

inline void
GlossyReflector::set_exponent(const float exp) {
	glossy_specular_brdf->mExponent = exp;
}

class Transparent : public Phong
{
public:
	Transparent();
	Transparent(float ior, float kr, float kt);
	Colour shade(ShadeRec& sr) override;
	void setSampler(std::shared_ptr<Sampler> sampler);

	void setIor(float ior);
	void setKr(float kr);
	void setKt(float kt);

	void setColour(Colour color);
private:
	std::shared_ptr<PerfectSpecular> mReflectiveBRDF;
	std::shared_ptr<PerfectTransmitter> mSpecularBTDF;

};