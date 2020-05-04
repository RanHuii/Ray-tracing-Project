#include "assignment.hpp"
#include "math.h"

// ******* Function Member Implementation *******
Colour checkOutOfGamut(Colour col) {
	float max = std::max(col.r, std::max(col.g, col.b));
	if (max > 1.0f)
	{
		return col / max;
	}
	else
		return col;

}
// ***** Shape function members *****
Shape::Shape() : mColour{ 0, 0, 0 }
{}

void Shape::setColour(Colour const& col)
{
	mColour = col;
}

Colour Shape::getColour() const
{
	return mColour;
}

void Shape::setMaterial(std::shared_ptr<Material> const& material)
{
	mMaterial = material;
}

std::shared_ptr<Material> Shape::getMaterial() const
{
	return mMaterial;
}

// ***** Sampler function members *****
Sampler::Sampler(int numSamples, int numSets) :
	mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 }, mCountDisk{ 0 }, mCountHemisphere{ 0 }
{
	mSamples.reserve(mNumSets* mNumSamples);
	mHemisphereSample.reserve(mNumSets* mNumSamples);
	disk_samples.reserve(mNumSets* mNumSamples);
	setupShuffledIndeces();

}

Sampler&
Sampler::operator= (const Sampler& rhs) {
	if (this == &rhs)
		return (*this);

	mNumSamples = rhs.mNumSamples;
	mNumSets = rhs.mNumSets;
	mSamples = rhs.mSamples;
	mShuffledIndeces = rhs.mShuffledIndeces;
	disk_samples = rhs.disk_samples;
	mHemisphereSample = rhs.mHemisphereSample;
	mCount = rhs.mCount;
	mJump = rhs.mJump;

	return (*this);
}


int Sampler::getNumSamples() const
{
	return mNumSamples;
}

void Sampler::setupShuffledIndeces()
{
	mShuffledIndeces.reserve(mNumSamples * mNumSets);
	std::vector<int> indices;

	std::random_device d;
	std::mt19937 generator(d());

	for (int j = 0; j < mNumSamples; ++j)
	{
		indices.push_back(j);
	}

	for (int p = 0; p < mNumSets; ++p)
	{
		std::shuffle(indices.begin(), indices.end(), generator);

		for (int j = 0; j < mNumSamples; ++j)
		{
			mShuffledIndeces.push_back(indices[j]);
		}
	}
}

atlas::math::Point Sampler::sampleUnitSquare()
{
	if (mCount % mNumSamples == 0)
	{
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}

	return mSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}

void Sampler::map_samples_to_disk()
{
	size_t size = mSamples.size();
	float r, phi;
	atlas::math::Point2 sp;
	disk_samples.reserve(size);

	for (size_t i = 0; i < size; i++)
	{
		sp.x = 2.0f * mSamples[i].x - 1.0f;
		sp.y = 2.0f * mSamples[i].y - 1.0f;

		if (sp.x > -sp.y)
		{
			if (sp.x > sp.y)
			{
				r = sp.x;
				phi = sp.y / sp.x;
			}
			else
			{
				r = sp.y;
				phi = 2 - sp.x / sp.y;
			}
		}
		else
		{
			if (sp.x < sp.y)
			{
				r = -sp.x;
				phi = 4 + sp.y / sp.x;
			}
			else
			{
				r = -sp.y;
				if (sp.y != 0.0)
					phi = 6 - sp.x / sp.y;
				else
					phi = 0.0;
			}
		}
		phi *= glm::pi<float>() / 4.0f;

		disk_samples.push_back(atlas::math::Point(r * glm::cos(phi), r * glm::sin(phi), 0));
		/*disk_samples[i].x = r * glm::cos(phi);
		disk_samples[i].y =r * glm::sin(phi);*/
	}


}

void Sampler::map_samples_to_hemisphere(const float e)
{
	mHemisphereSample.clear();
	size_t size = mSamples.size();
	mHemisphereSample.reserve(size);

	for (size_t i = 0; i < size; i++)
	{
		float cos_phi = glm::cos(2.0f * glm::pi<float>() * mSamples[i].x);
		float sin_phi = glm::sin(2.0f * glm::pi<float>() * mSamples[i].x);
		float cos_theta = pow((1.0f - mSamples[i].y), 1.0f / (e + 1.0f));
		float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
		float pu = sin_theta * cos_phi;
		float pv = sin_theta * sin_phi;
		float pw = cos_theta;
		mHemisphereSample.push_back(atlas::math::Point(pu, pv, pw));
	}
}

atlas::math::Point Sampler::sample_unit_hemisphere()
{
	if (mCountHemisphere % mNumSamples == 0)
	{
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}

	return mHemisphereSample[mJump + mShuffledIndeces[mJump + mCountHemisphere++ % mNumSamples]];
}

atlas::math::Point Sampler::sample_unit_disk()
{
	if (mCountDisk % mNumSamples == 0)
	{
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}

	return disk_samples[mJump + mShuffledIndeces[mJump + mCountDisk++ % mNumSamples]];
}

// ***** Light function members *****
Colour Light::L([[maybe_unused]] ShadeRec& sr)
{
	return mRadiance * mColour;
}

void Light::scaleRadiance(float b)
{
	mRadiance = b;
}

void Light::setColour(Colour const& c)
{
	mColour = c;
}
// ***** Triangle function members *****

Triangle::Triangle()
{
	v0 = atlas::math::Point(0, 0, 0);
	v1 = atlas::math::Point(0, 0, 1);
	v2 = atlas::math::Point(1, 0, 0);
	normal = atlas::math::Vector(0, 1, 0);
	v0v1 = v1 - v0;
	v0v2 = v2 - v0;

}
Triangle::Triangle(atlas::math::Point a, atlas::math::Point b, atlas::math::Point c)
{
	v0 = a;
	v1 = b;
	v2 = c;
	v0v1 = v1 - v0;
	v0v2 = v2 - v0;
	normal = glm::cross(v0v1, v0v2);

}
bool Triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	double kEpsilon = 0.01;
	float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
	float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
	float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

	float m = f * k - g * j, n = g * l - h * k, p = h * j - f * l;
	float q = g * i - e * k, s = e * j - f * i;

	float inv_denom = 1.0f / (a * m + b * q + c * s);

	float e1 = d * m - b * n - c * p;
	float beta = e1 * inv_denom;

	if (beta < 0.0f)
		return false;

	float r = e * l - h * i;
	float e2 = a * n + d * q + c * r;
	float gamma = e2 * inv_denom;

	if (gamma < 0.0f)
		return false;

	if (beta + gamma > 1.0f)
		return false;

	float e3 = a * p - b * r + (float)d * s;
	float t = e3 * inv_denom;

	if (t < kEpsilon)
		return false;

	tMin = t;

	return true;
}
bool Triangle::shadow_hit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const
{
	double kEpsilon = 0.01;
	float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
	float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
	float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

	float m = f * k - g * j, n = g * l - h * k, p = h * j - f * l;
	float q = g * i - e * k, s = e * j - f * i;

	float inv_denom = 1.0f / (a * m + b * q + c * s);

	float e1 = d * m - b * n - c * p;
	float beta = e1 * inv_denom;

	if (beta < 0.0f)
		return false;

	float r = e * l - h * i;
	float e2 = a * n + d * q + c * r;
	float gamma = e2 * inv_denom;

	if (gamma < 0.0f)
		return false;

	if (beta + gamma > 1.0f)
		return false;

	float e3 = a * p - b * r + d * s;
	float t = e3 * inv_denom;

	if (t < kEpsilon)
		return false;

	tmin = t;

	return true;
}

bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{

	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };
	if (intersect && t < sr.t)
	{
		sr.normal = glm::normalize(glm::cross((v1 - v0), (v2 - v0)));
		sr.ray = ray;
		sr.hit_point = ray.o + t * ray.d;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}
	return  intersect;
}

// ***** Sphere function members *****
Sphere::Sphere(atlas::math::Point center, float radius) :
	mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
{}

bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	atlas::math::Vector tmp = ray.o - mCentre;
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.t)
	{
		sr.normal = (tmp + t * ray.d) / mRadius;
		sr.ray = ray;
		sr.hit_point = sr.ray.o + t * sr.ray.d;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}

	return intersect;
}
bool Sphere::shadow_hit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
{
	const auto tmp{ ray.o - mCentre };
	const auto a{ glm::dot(ray.d, ray.d) };
	const auto b{ 2.0f * glm::dot(ray.d, tmp) };
	const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
	const auto disc{ (b * b) - (4.0f * a * c) };

	if (atlas::core::geq(disc, 0.0f))
	{
		const float kEpsilon{ 0.01f };
		const float e{ std::sqrt(disc) };
		const float denom{ 2.0f * a };


		float tsmall = (-b - e) / denom;
		float tbig = (-b + e) / denom;
		if (tsmall > kEpsilon)
		{
			tMin = tsmall;
			return true;
		}
		if (tbig > kEpsilon)
		{
			tMin = tbig;
			return true;
		}

	}
	return false;
}
bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const auto tmp{ ray.o - mCentre };
	const auto a{ glm::dot(ray.d, ray.d) };
	const auto b{ 2.0f * glm::dot(ray.d, tmp) };
	const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
	const auto disc{ (b * b) - (4.0f * a * c) };

	if (atlas::core::geq(disc, 0.0f))
	{
		const float kEpsilon{ 0.01f };
		const float e{ std::sqrt(disc) };
		const float denom{ 2.0f * a };

		// Look at the negative root first
		float t = (-b - e) / denom;
		if (atlas::core::geq(t, kEpsilon))
		{
			tMin = t;
			return true;
		}

		// Now the positive root
		t = (-b + e) / denom;
		if (atlas::core::geq(t, kEpsilon))
		{
			tMin = t;
			return true;
		}
	}

	return false;
}

Plane::Plane(atlas::math::Point point, atlas::math::Vector normal) :
	mpoint{ point }, mnormal{ normal }
{}

bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray,t) };
	if (intersect && t < sr.t)
	{
		sr.normal = mnormal;
		sr.ray = ray;
		sr.hit_point = sr.ray.o + t * sr.ray.d;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}
	return intersect;
}

bool Plane::shadow_hit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const
{
	float t = glm::dot((mpoint - ray.o), mnormal) / glm::dot(ray.d, mnormal);

	float kEpsilon = 0.01f;

	if (t > kEpsilon)
	{
		tmin = t;
		return true;
	}
	else
		return false;
}

bool Plane::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	float t = glm::dot((mpoint - ray.o), mnormal) / glm::dot(ray.d, mnormal);
	const float kEpsilon{ 0.01f };
	if (t > kEpsilon)
	{
		tMin = t;
		return true;
	}
	else
		return false;
};

// ***** BBox ******
// --------------------------------------------------------------------- default constructor

BBox::BBox(void)
	: x0(-1), x1(1), y0(-1), y1(1), z0(-1), z1(1)
{}


// --------------------------------------------------------------------- constructor

BBox::BBox(const double _x0, const double _x1,
	const double _y0, const double _y1,
	const double _z0, const double _z1)
	: x0(_x0), x1(_x1), y0(_y0), y1(_y1), z0(_z0), z1(_z1)
{}


// --------------------------------------------------------------------- constructor

BBox::BBox(const atlas::math::Point p0, const atlas::math::Point p1)
	: x0(p0.x), x1(p1.x), y0(p0.y), y1(p1.y), z0(p0.z), z1(p1.z)
{}



// --------------------------------------------------------------------- copy constructor

BBox::BBox(const BBox& bbox)
	: x0(bbox.x0), x1(bbox.x1), y0(bbox.y0), y1(bbox.y1), z0(bbox.z0), z1(bbox.z1)
{}


// --------------------------------------------------------------------- assignment operator

BBox&
BBox::operator= (const BBox& rhs) {
	if (this == &rhs)
		return (*this);

	x0 = rhs.x0;
	x1 = rhs.x1;
	y0 = rhs.y0;
	y1 = rhs.y1;
	z0 = rhs.z0;
	z1 = rhs.z1;

	return (*this);
}


// --------------------------------------------------------------------- destructor

BBox::~BBox(void) {}


// --------------------------------------------------------------------- hit

bool
BBox::hit(atlas::math::Ray<atlas::math::Vector> const& ray) const {

	double kEpsilon = 0.001;
	double ox = ray.o.x; double oy = ray.o.y; double oz = ray.o.z;
	double dx = ray.d.x; double dy = ray.d.y; double dz = ray.d.z;

	double tx_min, ty_min, tz_min;
	double tx_max, ty_max, tz_max;

	double a = 1.0 / dx;
	if (a >= 0) {
		tx_min = (x0 - ox) * a;
		tx_max = (x1 - ox) * a;
	}
	else {
		tx_min = (x1 - ox) * a;
		tx_max = (x0 - ox) * a;
	}

	double b = 1.0 / dy;
	if (b >= 0) {
		ty_min = (y0 - oy) * b;
		ty_max = (y1 - oy) * b;
	}
	else {
		ty_min = (y1 - oy) * b;
		ty_max = (y0 - oy) * b;
	}

	double c = 1.0 / dz;
	if (c >= 0) {
		tz_min = (z0 - oz) * c;
		tz_max = (z1 - oz) * c;
	}
	else {
		tz_min = (z1 - oz) * c;
		tz_max = (z0 - oz) * c;
	}

	double t0, t1;

	// find largest entering t value

	if (tx_min > ty_min)
		t0 = tx_min;
	else
		t0 = ty_min;

	if (tz_min > t0)
		t0 = tz_min;

	// find smallest exiting t value

	if (tx_max < ty_max)
		t1 = tx_max;
	else
		t1 = ty_max;

	if (tz_max < t1)
		t1 = tz_max;

	return (t0 < t1 && t1 > kEpsilon);
}


// --------------------------------------------------------------------- inside
// used to test if a ray starts inside a grid

bool
BBox::inside(const atlas::math::Point& p) const {
	return ((p.x > x0 && p.x < x1) && (p.y > y0 && p.y < y1) && (p.z > z0 && p.z < z1));
};
//***** Box ******
Box::Box(atlas::math::Point closeCornor, atlas::math::Point farCornor)
{
	this->closeCornor = closeCornor;
	this->farCornor = farCornor;
}

bool Box::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray,t) };
	if (intersect && t < sr.t)
	{
		sr.ray = ray;
		sr.hit_point = sr.ray.o + t * sr.ray.d;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}
	return intersect;
}

bool Box::shadow_hit(atlas::math::Ray<atlas::math::Vector> const& ray, float& t) const
{

	float ox = ray.o.x;
	float oy = ray.o.y;
	float oz = ray.o.z;

	float dx = ray.d.x;
	float dy = ray.d.y;
	float dz = ray.d.z;

	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;

	float a = 1.0f / dx;
	float b = 1.0f / dy;
	float c = 1.0f / dz;

	if (a >= 0.0f)
	{
		tx_min = (closeCornor.x - ox) * a;
		tx_max = (farCornor.x - ox) * a;
	}
	else
	{
		tx_min = (farCornor.x - ox) * a;
		tx_max = (closeCornor.x - ox) * a;
	}

	if (b >= 0.0f)
	{
		ty_min = (closeCornor.y - oy) * b;
		ty_max = (farCornor.y - oy) * b;
	}
	else
	{
		ty_min = (farCornor.y - oy) * b;
		ty_max = (closeCornor.y - oy) * b;
	}
	if (tx_min > ty_max || ty_min > tx_max)
	{
		return false;
	}
	if (ty_min > tx_min)
	{
		tx_min = ty_min;
	}
	if (ty_max < tx_max)
	{
		tx_max = ty_max;
	}

	if (c >= 0.0f)
	{
		tz_min = (closeCornor.z - oz) * c;
		tz_max = (farCornor.z - oz) * c;
	}
	else
	{
		tz_min = (farCornor.z - oz) * c;
		tz_max = (closeCornor.z - oz) * c;
	}

	if ((tx_min > tz_max) || (tz_min > tx_max))
		return false;
	if (tz_min > tx_min)
		tx_min = tz_min;
	if (tz_max < tx_max)
		tx_max = tz_max;

	t = tx_min;
	return true;
}

bool Box::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
{
	float ox = ray.o.x;
	float oy = ray.o.y;
	float oz = ray.o.z;

	float dx = ray.d.x;
	float dy = ray.d.y;
	float dz = ray.d.z;

	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;

	float a = 1.0f / dx;
	float b = 1.0f / dy;
	float c = 1.0f / dz;

	if (a >= 0.0f)
	{
		tx_min = (closeCornor.x - ox) * a;
		tx_max = (farCornor.x - ox) * a;
	}
	else
	{
		tx_min = (farCornor.x - ox) * a;
		tx_max = (closeCornor.x - ox) * a;
	}

	if (b >= 0.0f)
	{
		ty_min = (closeCornor.y - oy) * b;
		ty_max = (farCornor.y - oy) * b;
	}
	else
	{
		ty_min = (farCornor.y - oy) * b;
		ty_max = (closeCornor.y - oy) * b;
	}
	if (tx_min > ty_max || ty_min > tx_max)
	{
		return false;
	}
	if (ty_min > tx_min)
	{
		tx_min = ty_min;
	}
	if (ty_max < tx_max)
	{
		tx_max = ty_max;
	}

	if (c >= 0.0f)
	{
		tz_min = (closeCornor.z - oz) * c;
		tz_max = (farCornor.z - oz) * c;
	}
	else
	{
		tz_min = (farCornor.z - oz) * c;
		tz_max = (closeCornor.z - oz) * c;
	}

	if ((tx_min > tz_max) || (tz_min > tx_max))
		return false;
	if (tz_min > tx_min)
		tx_min = tz_min;
	if (tz_max < tx_max)
		tx_max = tz_max;


	tMin = tx_min;
	return true;
}

// ***** Rectangle class ******

const double Rectangle::kEpsilon = 0.001;



// ----------------------------------------------------------------  constructor
// this constructs the normal

Rectangle::Rectangle(const atlas::math::Point& _p0, const atlas::math::Vector& _a, const atlas::math::Vector& _b)
	: Shape(),
	p0(_p0),
	a(_a),
	b(_b),
	a_len_squared(glm::pow(glm::length(a), 2)),
	b_len_squared(glm::pow(glm::length(b), 2)),
	area(glm::length(a)* glm::length(b)),
	inv_area(1.0f / area),
	sampler_ptr(NULL)
{
	normal = glm::cross(a, b);
	glm::normalize(normal);
}


// ----------------------------------------------------------------  constructor
// this has the normal as an argument

Rectangle::Rectangle(const atlas::math::Point& _p0, const atlas::math::Vector& _a, const atlas::math::Vector& _b, const atlas::math::Normal& n)
	: Shape(),
	p0(_p0),
	a(_a),
	b(_b),
	a_len_squared(glm::pow(glm::length(a), 2)),
	b_len_squared(glm::pow(glm::length(b), 2)),
	area(glm::length(a)* glm::length(b)),
	inv_area(1.0f / area),
	normal(n),
	sampler_ptr(NULL)
{
	glm::normalize(normal);
}


//------------------------------------------------------------------ get_bounding_box 

BBox
Rectangle::get_bounding_box(void) {
	double delta = 0.0001;

	return(BBox(glm::min(p0.x, p0.x + a.x + b.x) - delta, glm::max(p0.x, p0.x + a.x + b.x) + delta,
		glm::min(p0.y, p0.y + a.y + b.y) - delta, glm::max(p0.y, p0.y + a.y + b.y) + delta,
		glm::min(p0.z, p0.z + a.z + b.z) - delta, glm::max(p0.z, p0.z + a.z + b.z) + delta));
}


//------------------------------------------------------------------ hit 

bool
Rectangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {

	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };
	if (intersect && t < sr.t)
	{
		sr.normal = normal;
		sr.ray = ray;
		sr.hit_point = ray.o + t * ray.d;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}
	return  intersect;


}

bool Rectangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	float t = glm::dot((p0 - ray.o), normal) / glm::dot(ray.d, normal);

	if (t <= kEpsilon)
		return (false);

	atlas::math::Point p = ray.o + t * ray.d;
	atlas::math::Vector d = p - p0;

	float ddota = glm::dot(d, a);

	if (ddota < 0.0 || ddota > a_len_squared)
		return (false);

	float ddotb = glm::dot(d, b);

	if (ddotb < 0.0 || ddotb > b_len_squared)
		return (false);

	tMin = t;
	return (true);
}

// ---------------------------------------------------------------- setSampler

void
Rectangle::set_sampler(std::shared_ptr<Sampler> sampler) {
	sampler_ptr = sampler;
}


// ---------------------------------------------------------------- sample
// returns a sample point on the rectangle

atlas::math::Point
Rectangle::sample(void) {
	atlas::math::Point2 sample_point = sampler_ptr->sampleUnitSquare();
	return (p0 + sample_point.x * a + sample_point.y * b);
}


//------------------------------------------------------------------ get_normal 

atlas::math::Normal
Rectangle::get_normal([[maybe_unused]] const atlas::math::Point& p) {
	return (normal);
}


// ---------------------------------------------------------------- pdf

float
Rectangle::pdf([[maybe_unused]] ShadeRec& sr) {
	return (inv_area);
}


Camera::Camera() :
	mEye{ 0.0f, 0.0f, 500.0f },
	mLookAt{ 0.0f },
	mUp{ 0.0f, 1.0f, 0.0f },
	mU{ 1.0f, 0.0f, 0.0f },
	mV{ 0.0f, 1.0f, 0.0f },
	mW{ 0.0f, 0.0f, 1.0f }
{}

void Camera::setEye(atlas::math::Point const& eye)
{
	mEye = eye;
}

void Camera::setLookAt(atlas::math::Point const& lookAt)
{
	mLookAt = lookAt;
}

void Camera::setUpVector(atlas::math::Vector const& up)
{
	mUp = up;
}

void Camera::computeUVW()
{
	mW = glm::normalize(mEye - mLookAt);
	mU = glm::normalize(glm::cross(mUp, mW));
	mV = glm::cross(mW, mU);

	if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
		mEye.y > mLookAt.y)
	{
		mU = { 0.0f, 0.0f, 1.0f };
		mV = { 1.0f, 0.0f, 0.0f };
		mW = { 0.0f, 1.0f, 0.0f };
	}

	if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
		mEye.y < mLookAt.y)
	{
		mU = { 1.0f, 0.0f, 0.0f };
		mV = { 0.0f, 0.0f, 1.0f };
		mW = { 0.0f, -1.0f, 0.0f };
	}
}

// ***** Pinhole function members *****
Pinhole::Pinhole() : Camera{}, mDistance{ 500.0f }, mZoom{ 1.0f }
{}

void Pinhole::setDistance(float distance)
{
	mDistance = distance;
}

void Pinhole::setZoom(float zoom)
{
	mZoom = zoom;
}

atlas::math::Vector Pinhole::rayDirection(atlas::math::Point const& p) const
{
	const auto dir = p.x * mU + p.y * mV - mDistance * mW;
	return glm::normalize(dir);
}

void Pinhole::renderScene(std::shared_ptr<World> world) const
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	Point samplePoint{}, pixelPoint{};
	Ray<atlas::math::Vector> ray{};

	ray.o = mEye;
	float avg{ 1.0f / world->sampler->getNumSamples() };

	for (int r{ 0 }; r < world->height; ++r)
	{
		for (int c{ 0 }; c < world->width; ++c)
		{
			Colour pixelAverage{ 0, 0, 0 };

			for (int j = 0; j < world->sampler->getNumSamples(); ++j)
			{
				ShadeRec trace_data{};
				trace_data.world = world;
				trace_data.t = std::numeric_limits<float>::max();
				trace_data.depth = 0;
				samplePoint = world->sampler->sampleUnitSquare();
				pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
				pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
				ray.d = rayDirection(pixelPoint);
				bool hit{};

				for (auto obj : world->scene)
				{
					hit |= obj->hit(ray, trace_data);
				}

				if (hit)
				{
					pixelAverage += trace_data.material->shade(trace_data);
				}
			}
			Colour temp = Colour(pixelAverage.r * avg,
				pixelAverage.g * avg,
				pixelAverage.b * avg);
			world->image.push_back(checkOutOfGamut(temp));
		}
	}
}


Jittered::Jittered(int numSamples, int numSets) :Sampler{ numSamples, numSets }
{
	generateSamples();
	map_samples_to_hemisphere(1);

}

void Jittered::generateSamples()
{
	atlas::math::Random<float> engine;
	int n = (int)sqrt(mNumSamples);
	for (int p = 0; p < mNumSets; p++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				atlas::math::Point sp((k + engine.getRandomOne()) / n, (j + engine.getRandomOne()) / n, 0.0f);
				mSamples.push_back(sp);
			}
		}
	}
}

Jittered&
Jittered::operator= (const Jittered& rhs) {
	if (this == &rhs)
		return (*this);

	Sampler::operator=(rhs);

	return (*this);
}


MultiJittered&
MultiJittered::operator= (const MultiJittered& rhs) {
	if (this == &rhs)
		return (*this);

	Sampler::operator=(rhs);

	return (*this);
}
// ---------------------------------------------------------------- constructor

MultiJittered::MultiJittered(const int num_samples, const int m)
	: Sampler(num_samples, m) {
	generateSamples();
	map_samples_to_hemisphere(1);
}



void
MultiJittered::generateSamples(void) {
	// num_samples needs to be a perfect square

	int n = (int)sqrt(mNumSamples);
	float subcell_width = 1.0f / ((float)mNumSamples);

	// fill the samples array with dummy points to allow us to use the [ ] notation when we set the 
	// initial patterns

	atlas::math::Point fill_point;
	for (int j = 0; j < mNumSamples * mNumSets; j++)
		mSamples.push_back(fill_point);

	// distribute points in the initial patterns
	atlas::math::Random<float> engine;
	engine.getRandomRange(0, subcell_width);
	for (int p = 0; p < mNumSets; p++)
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				mSamples[i * n + j + p * mNumSamples].x = (i * n + j) * subcell_width + engine.getRandomRange(0, subcell_width);
				mSamples[i * n + j + p * mNumSamples].y = (j * n + i) * subcell_width + engine.getRandomRange(0, subcell_width);
				mSamples[i * n + j + p * mNumSamples].z = 0.0f;
			}

	// shuffle x coordinates

	for (int p = 0; p < mNumSets; p++)
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				int k = j + (rand() % static_cast<int>(n - 1 - j + 1));
				float t = mSamples[i * n + j + p * mNumSamples].x;
				mSamples[i * n + j + p * mNumSamples].x = mSamples[i * n + k + p * mNumSamples].x;
				mSamples[i * n + k + p * mNumSamples].x = t;
			}

	// shuffle y coordinates

	for (int p = 0; p < mNumSets; p++)
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				int k = j + (rand() % static_cast<int>(n - 1 - j + 1));
				float t = mSamples[j * n + i + p * mNumSamples].y;
				mSamples[j * n + i + p * mNumSamples].y = mSamples[k * n + i + p * mNumSamples].y;
				mSamples[k * n + i + p * mNumSamples].y = t;
			}
}

// ***** Lambertian function members *****
Lambertian::Lambertian() : mDiffuseColour{}, mDiffuseReflection{}
{}

Lambertian::Lambertian(Colour diffuseColor, float diffuseReflection) :
	mDiffuseColour{ diffuseColor }, mDiffuseReflection{ diffuseReflection }
{}

Colour
Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();
}

Colour
Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return mDiffuseColour * mDiffuseReflection;
}

void Lambertian::setDiffuseReflection(float kd)
{
	mDiffuseReflection = kd;
}

void Lambertian::setDiffuseColour(Colour const& colour)
{
	mDiffuseColour = colour;
}
// ***** GlossySpecular members *****
GlossySpecular::GlossySpecular(void) :
	mSpecularColour{}, mDiffuseReflection{}, mSpecularReflection{}, mExponent{}
{}


GlossySpecular::GlossySpecular(Colour specularColour, float diffuseReflection,
	float specularReflection, float exponent) :

	mSpecularColour{ specularColour }, mDiffuseReflection{ diffuseReflection },
	mSpecularReflection{ specularReflection }, mExponent{ exponent }
{}

// ---------------------------------------------------------------------- destructor

GlossySpecular::~GlossySpecular(void) {}


// ---------------------------------------------------------------------- clone

GlossySpecular*
GlossySpecular::clone(void) const {
	return (new GlossySpecular(*this));
}


// ---------------------------------------------------------------------- set_sampler
// this allows any type of sampling to be specified in the build functions

void
GlossySpecular::set_sampler(std::shared_ptr<Sampler> sp) {
	sampler_ptr = sp;
}

void
GlossySpecular::set_sampler(const int num_samples, const int num_sets)
{
	mSampler = std::make_shared<MultiJittered>(num_samples, num_sets);
	mSampler->map_samples_to_hemisphere(mExponent);
}

// ----------------------------------------------------------------------------------- f
// no sampling here: just use the Phong formula
// this is used for direct illumination only

Colour
GlossySpecular::fn(const ShadeRec& sr, const atlas::math::Vector& wo, const atlas::math::Vector& wi) const {
	Colour 	L = Colour(0, 0, 0);
	float ndotwi = glm::dot(sr.normal, wi);
	atlas::math::Vector r(-wi + 2.0f * sr.normal * ndotwi); // mirror reflection direction
	float rdotwo = glm::dot(r, wo);

	if (rdotwo > 0.0)
		L = mSpecularReflection * glm::pow(rdotwo, mExponent) * mSpecularColour;

	return (L);
}


// ----------------------------------------------------------------------------------- sample_f
// this is used for indirect illumination

Colour
GlossySpecular::sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wi, float& pdf) const {

	float ndotwo = glm::dot(sr.normal, wo);
	atlas::math::Vector r = -wo + 2.0f * sr.normal * ndotwo;

	atlas::math::Vector w = r;
	atlas::math::Vector u = glm::normalize(glm::cross(atlas::math::Vector(0.00424, 1, 0.00764), w));
	atlas::math::Vector v = glm::cross(u, w);

	atlas::math::Point sp = mSampler->sample_unit_hemisphere();
	wi = sp.x * u + sp.y * v + sp.z * w;

	if (glm::dot(sr.normal, wi) < 0.0f)
	{
		wi = -sp.x * u - sp.y * v + sp.z * w;
	}
	float phong_lobe = pow(glm::dot(r, wi), mExponent);
	pdf = phong_lobe * glm::dot(sr.normal, wi);

	return (kr * mSpecularColour * phong_lobe);
}


// ----------------------------------------------------------------------------------- rho

Colour
GlossySpecular::rho([[maybe_unused]] const ShadeRec& sr, [[maybe_unused]] const atlas::math::Vector& wo) const {
	return (Colour(0, 0, 0));
}

void GlossySpecular::setDiffuseReflection(float kd)
{
	mDiffuseReflection = kd;
}

void GlossySpecular::setSpecularReflection(float ks)
{
	mSpecularReflection = ks;
}

void GlossySpecular::setSpecularColour(Colour const& colour)
{
	mSpecularColour = colour;
}

void GlossySpecular::setKr(float kr1)
{
	kr = kr1;
}

void GlossySpecular::setExponent(float exp)
{
	mExponent = exp;
}


// **** PerfectSpecular *****
PerfectSpecular::PerfectSpecular(void)
	: BRDF(),
	kr(0.0),
	cr(1.0)
{}

// ---------------------------------------------------------- destructor

PerfectSpecular::~PerfectSpecular(void) {}


// ---------------------------------------------------------------------- clone

PerfectSpecular*
PerfectSpecular::clone(void) const {
	return (new PerfectSpecular(*this));
}


// ---------------------------------------------------------- f

Colour
PerfectSpecular::fn([[maybe_unused]] const ShadeRec& sr, [[maybe_unused]] const atlas::math::Vector& wo, [[maybe_unused]] const atlas::math::Vector& wi) const {
	return (Colour(0, 0, 0));
}
void
PerfectSpecular::set_sampler(std::shared_ptr<Sampler> sampler)
{
	mSampler = sampler;
	mSampler->map_samples_to_hemisphere(1);
}

// ---------------------------------------------------------- sample_f
// this computes wi: the direction of perfect mirror reflection
// it's called from from the functions Reflective::shade and Transparent::shade.
// the fabs in the last statement is for transparency

Colour
PerfectSpecular::sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wi) const {
	float ndotwo = glm::dot(sr.normal, wo);
	wi = -wo + 2.0f * sr.normal * ndotwo;
	return (kr * cr / glm::abs(glm::dot(sr.normal, wi))); // why is this fabs? // kr would be a Fresnel term in a Fresnel reflector
}											 // for transparency when ray hits inside surface?, if so it should go in Chapter 24


// ---------------------------------------------------------- sample_f
// this version of sample_f is used with path tracing
// it returns ndotwi in the pdf

Colour
PerfectSpecular::sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wi, float& pdf) const {
	float ndotwo = glm::dot(sr.normal, wo);
	wi = -wo + 2.0f * sr.normal * ndotwo;
	pdf = glm::abs(glm::dot(sr.normal, wi));
	return (kr * cr);
}

// ---------------------------------------------------------- rho

Colour
PerfectSpecular::rho([[maybe_unused]] const ShadeRec& sr, [[maybe_unused]] const atlas::math::Vector& wo) const {
	return (Colour(0, 0, 0));
}


// ***** Matte function members *****
Matte::Matte() :
	Material{},
	mDiffuseBRDF{ std::make_shared<Lambertian>() },
	mAmbientBRDF{ std::make_shared<Lambertian>() }
{}

Matte::Matte(float kd, float ka, Colour color) : Matte{}
{
	setDiffuseReflection(kd);
	setAmbientReflection(ka);
	setDiffuseColour(color);
}

void Matte::setDiffuseReflection(float k)
{
	mDiffuseBRDF->setDiffuseReflection(k);
}

void Matte::setAmbientReflection(float k)
{
	mAmbientBRDF->setDiffuseReflection(k);
}

void Matte::setDiffuseColour(Colour colour)
{
	mDiffuseBRDF->setDiffuseColour(colour);
	mAmbientBRDF->setDiffuseColour(colour);
}

Colour Matte::shade(ShadeRec& sr)
{
	using atlas::math::Ray;
	using atlas::math::Vector;

	Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
	size_t numLights = sr.world->lights.size();

	for (size_t i{ 0 }; i < numLights; ++i)
	{
		Vector wi = sr.world->lights[i]->getDirection(sr);
		float nDotWi = glm::dot(sr.normal, wi);

		if (nDotWi > 0.0f)
		{
			bool in_shadow = false;

			if (sr.world->lights[i]->cast_shadows())
			{
				Ray shadowRay(sr.hit_point, wi);
				in_shadow = sr.world->lights[i]->in_shadow(shadowRay, sr);
			}
			if (!in_shadow)
			{
				L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) *
					nDotWi;
			}

		}
	}

	return L;
}
// **** Phong ***** 
Phong::Phong(void) :
	Material{},
	diffuse_brdf{ std::make_shared<Lambertian>() },
	ambient_brdf{ std::make_shared<Lambertian>() }

{
	specular_brdf = new GlossySpecular();
}

Phong::Phong(float kd, float ka, Colour color) : Phong{}
{
	setDiffuseReflection(kd);
	setAmbientReflection(ka);
	setDiffuseColour(color);
	specular_brdf->mSpecularColour = Colour(1, 1, 1);
	specular_brdf->mExponent = 100;
	specular_brdf->mSpecularReflection = 0.1f;
	specular_brdf->set_sampler(64, 100);
}

void Phong::setDiffuseReflection(float k)
{
	diffuse_brdf->setDiffuseReflection(k);
}

void Phong::setAmbientReflection(float k)
{
	ambient_brdf->setDiffuseReflection(k);
}

void Phong::setDiffuseColour(Colour colour)
{
	diffuse_brdf->setDiffuseColour(colour);
	ambient_brdf->setDiffuseColour(colour);
}

Colour
Phong::shade(ShadeRec& sr) {
	atlas::math::Vector wo = -sr.ray.d;
	Colour 	L = ambient_brdf->rho(sr, wo) * sr.world->ambient->L(sr);
	size_t	num_lights = sr.world->lights.size();

	for (int j = 0; j < num_lights; j++) {
		Colour wi = sr.world->lights[j]->getDirection(sr);
		float ndotwi = glm::dot(sr.normal, wi);

		if (ndotwi > 0.0)
			L += (diffuse_brdf->fn(sr, wo, wi) +
				specular_brdf->fn(sr, wo, wi)) * sr.world->lights[j]->L(sr) * ndotwi;
	}

	return (L);
}

// ***** Ambient Occluder members *****
AmbientOccluder::AmbientOccluder()
{

}

AmbientOccluder::AmbientOccluder(float min)
{
	min_amount = min;
	count = 0;

}

void AmbientOccluder::set_sampler(std::shared_ptr<Sampler> s_ptr)
{

}

atlas::math::Vector AmbientOccluder::getDirection([[maybe_unused]] ShadeRec& sr)
{
	atlas::math::Point sp = sr.world->sampler->sample_unit_hemisphere();

	return sp.x * u + sp * v + sp.z * w;
}

bool AmbientOccluder::in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, const ShadeRec& sr) const
{
	float kEpsilon = 0.05f;
	float t = 0;
	for (auto obj : sr.world->scene)
	{
		if (obj->intersectRay(ray, t) && t > kEpsilon)
		{
			return true;
		}

	}
	return false;
}

bool AmbientOccluder::cast_shadows()
{
	return true;
}

Colour AmbientOccluder::L(ShadeRec& sr)
{
	w = sr.normal;

	// jitter up vector in case normal is vertical
	v = glm::cross(w, atlas::math::Vector(0.0072, 1.0, 0.0034));

	v = glm::normalize(v);

	u = glm::cross(v, w);

	atlas::math::Ray<atlas::math::Vector> shadow_ray{ sr.hit_point, getDirection(sr) };

	if (in_shadow(shadow_ray, sr))
	{
		return(min_amount * mRadiance * mColour);
	}
	else
	{
		return (mRadiance * mColour);
	}

}

// ***** Directional function members *****
Directional::Directional() : Light{}
{}

Directional::Directional(atlas::math::Vector const& d) : Light{}
{
	setDirection(d);
}

bool Directional::in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, const ShadeRec& sr) const
{
	float t;

	for (auto obj : sr.world->scene)
	{
		if (obj->shadow_hit(ray, t))
			return true;
	}
	return false;
}

bool Directional::cast_shadows()
{
	return false;
}

void Directional::setDirection(atlas::math::Vector const& d)
{
	mDirection = glm::normalize(d);
}

atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return mDirection;
}

// ***** Ambient function members *****
Ambient::Ambient() : Light{}
{}

atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return atlas::math::Vector{ 0.0f };
}

bool Ambient::cast_shadows()
{
	return false;
}

bool Ambient::in_shadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] const ShadeRec& sr) const
{
	return true;
}

PointLight::PointLight() :Light{}
{

}

void PointLight::setLocation(const atlas::math::Vector lo) {
	location = lo;
}

PointLight::PointLight(const atlas::math::Vector lo) : Light{}
{
	setLocation(lo);
}

bool PointLight::cast_shadows()
{
	return true;
}

bool PointLight::in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, const ShadeRec& sr) const {
	float t;

	float d = glm::distance(location, ray.o);

	for (auto obj : sr.world->scene)
	{
		if (obj->shadow_hit(ray, t) && t < d)
			return true;
	}
	return false;
}
atlas::math::Vector PointLight::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return glm::normalize((location - (sr.ray.d * sr.t)));
}

BTDF::BTDF(void) {}


// ------------------------------------------------------------------------ copy constructor

BTDF::BTDF([[maybe_unused]] const BTDF& btdf) {}


// ------------------------------------------------------------------------ destructor

BTDF::~BTDF(void) {}


// ------------------------------------------------------------------------ assignment operator

BTDF&
BTDF::operator= (const BTDF& rhs) {
	if (this == &rhs)
		return (*this);

	return (*this);
}


PerfectTransmitter::PerfectTransmitter(void)
	: BTDF(),
	kt(0.0),
	ior(1.0)
{}

PerfectTransmitter::PerfectTransmitter(float ior, float kt)
{
	this->ior = ior;
	this->kt = kt;
}


// ------------------------------------------------------------------- copy constructor

PerfectTransmitter::PerfectTransmitter(const PerfectTransmitter& pt)
	: BTDF(pt),
	kt(pt.kt),
	ior(pt.ior)
{}

void
PerfectTransmitter::set_sampler(std::shared_ptr<Sampler> sampler)
{
	mSampler = sampler;
	mSampler->map_samples_to_hemisphere(1);
}

void
PerfectTransmitter::set_colour(Colour c)
{
	mColour = c;
}
// ------------------------------------------------------------------- clone

PerfectTransmitter*
PerfectTransmitter::clone(void) {
	return (new PerfectTransmitter(*this));
}


// ------------------------------------------------------------------- destructor

PerfectTransmitter::~PerfectTransmitter(void) {}



// ------------------------------------------------------------------- assignment operator

PerfectTransmitter&
PerfectTransmitter::operator= (const PerfectTransmitter& rhs) {
	if (this == &rhs)
		return (*this);

	kt = rhs.kt;
	ior = rhs.ior;

	return (*this);
}


// ------------------------------------------------------------------- tir
// tests for total internal reflection
void
PerfectTransmitter::set_kt(const float k)
{
	kt = k;
}

void
PerfectTransmitter::set_ior(const float eta)
{
	ior = eta;
}

bool
PerfectTransmitter::tir(const ShadeRec& sr) const {
	atlas::math::Vector wo(-sr.ray.d);
	float cos_thetai = glm::dot(sr.normal, wo);
	float eta = ior;

	if (cos_thetai < 0.0f)
		eta = 1.0f / eta;

	return (1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta) < 0.0f);
}


// ------------------------------------------------------------------- f

Colour
PerfectTransmitter::f([[maybe_unused]] const ShadeRec& sr, [[maybe_unused]] const atlas::math::Vector& wo, [[maybe_unused]] const atlas::math::Vector& wi) const {
	return (Colour(0, 0, 0));
}


// ------------------------------------------------------------------- sample_f
// this computes the direction wt for perfect transmission
// and returns the transmission coefficient
// this is only called when there is no total internal reflection

Colour
PerfectTransmitter::sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wt) const {

	atlas::math::Vector n(sr.normal);
	float cos_thetai = glm::dot(n, wo);
	float eta = ior;

	if (cos_thetai < 0.0f) {			// transmitted ray is outside     
		cos_thetai = -cos_thetai;
		n = -n;  					// reverse direction of normal
		eta = 1.0f / eta; 			// invert ior 
	}

	float temp = 1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta);
	float cos_theta2 = sqrt(temp);
	wt = -wo / eta - (cos_theta2 - cos_thetai / eta) * n;

	return (kt / (eta * eta) * Colour(1, 1, 1) / glm::abs(glm::dot(sr.normal, wt)));
}


// ------------------------------------------------------------------- rho

Colour
PerfectTransmitter::rho([[maybe_unused]] const ShadeRec& sr, [[maybe_unused]] const atlas::math::Vector& wo) const {
	return (Colour(0, 0, 0));
}

// ******* tracer Code *******
Tracer::Tracer(void)
	: world_ptr(NULL)
{}


// -------------------------------------------------------------------- constructor

Tracer::Tracer(std::shared_ptr<World> _worldPtr)
	: world_ptr(_worldPtr)
{}



// -------------------------------------------------------------------- destructor

Tracer::~Tracer(void) {
	if (world_ptr)
		world_ptr = NULL;
}


// -------------------------------------------------------------------- trace_ray

Colour
Tracer::trace_ray([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray) const {
	return (Colour(0, 0, 0));
}


// -------------------------------------------------------------------- trace_ray

Colour
Tracer::trace_ray([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] const int depth) const {
	return (Colour(0, 0, 0));
}

// ******* Whitted tracer ******

Whitted::Whitted(void)
	: Tracer()
{}


// -------------------------------------------------------------------- constructor

Whitted::Whitted(std::shared_ptr<World> _worldPtr)
	: Tracer(_worldPtr)
{}


// -------------------------------------------------------------------- destructor

Whitted::~Whitted(void) {}


// -------------------------------------------------------------------- trace_ray

Colour
Whitted::trace_ray(atlas::math::Ray<atlas::math::Vector> const& ray, const int depth) const {

	if (depth > world_ptr->max_depth)
		return(Colour(0, 0, 0));
	else {
		ShadeRec sr{};
		sr.t = std::numeric_limits<float>::max();
		sr.world = world_ptr;
		bool hit = false;
		for (auto obj : world_ptr->scene)
		{
			if (obj->hit(ray, sr))
				hit = true;
		}
		if (hit)
		{
			sr.depth = depth;
			sr.ray = ray;
			return (sr.material->shade(sr));
		}
		else
			return (Colour(0, 0, 0));
	}
}

Reflective::Reflective(void)
	: Phong(),
	reflective_brdf(new PerfectSpecular)
{}

Reflective::Reflective(float kd, float ka, Colour color) :
	Phong(kd, ka, color), reflective_brdf(new PerfectSpecular)
{
	set_exp(1000);
	set_ks(0.35f);
	set_kr(0.9f);
	set_cr(Colour(1, 1, 1));
}
// ---------------------------------------------------------------- copy constructor

Reflective::Reflective(const Reflective& rm)
	: Phong(rm) {

	if (rm.reflective_brdf)
		reflective_brdf = rm.reflective_brdf->clone();
	else
		reflective_brdf = NULL;
}


// ------------------------------------------------------------------------------------ shade 

Colour
Reflective::shade(ShadeRec& sr) {
	Colour L(Phong::shade(sr));  // direct illumination

	atlas::math::Vector wo = -sr.ray.d;
	atlas::math::Vector wi;
	Colour fr = reflective_brdf->sample_f(sr, wo, wi);
	atlas::math::Ray<atlas::math::Vector> reflected_ray(sr.hit_point, wi);


	L += fr * sr.world->tracer_ptr->trace_ray(reflected_ray, sr.depth + 1) * (glm::dot(sr.normal, wi));

	return (L);
}

// ***** Regular function members *****
Regular::Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
	map_samples_to_hemisphere(1);
}

void Regular::generateSamples()
{
	int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

	for (int j = 0; j < mNumSets; ++j)
	{
		for (int p = 0; p < n; ++p)
		{
			for (int q = 0; q < n; ++q)
			{
				mSamples.push_back(
					atlas::math::Point{ (q + 0.5f) / n, (p + 0.5f) / n, 0.0f });
			}
		}
	}
}
// **** glossy reflector ******

GlossyReflector::GlossyReflector(void)
{
	glossy_specular_brdf = new GlossySpecular();
}

GlossyReflector::GlossyReflector([[maybe_unused]] float kd, [[maybe_unused]] float ka, [[maybe_unused]] Colour color) :
	Phong(kd, ka, color)
{
	glossy_specular_brdf = new GlossySpecular(Colour(1, 1, 1), 0.1f, 0.9f, 100);
	glossy_specular_brdf->set_sampler(64, 100);
	glossy_specular_brdf->kr = 0.8f;
}
Colour
GlossyReflector::shade(ShadeRec& sr) {
	Colour L(Phong::shade(sr));  // direct illumination

	atlas::math::Vector wo = -sr.ray.d;
	atlas::math::Vector wi;
	float pdf = 0.0f;
	Colour fr = glossy_specular_brdf->sample_f(sr, wo, wi, pdf);

	atlas::math::Ray<atlas::math::Vector> reflected_ray(sr.hit_point, wi);


	L += fr * sr.world->tracer_ptr->trace_ray(reflected_ray, sr.depth + 1) * (glm::dot(sr.normal, wi)) / pdf;


	return (L);
}

Transparent::Transparent() : mReflectiveBRDF{}, mSpecularBTDF{}
{
	mReflectiveBRDF = std::make_shared<PerfectSpecular>();
	mSpecularBTDF = std::make_shared<PerfectTransmitter>();
}

Transparent::Transparent(float ior, float kr, float kt) : Transparent()
{
	setIor(ior);
	setKr(kr);
	setKt(kt);
}
void Transparent::setColour(Colour color) {
	mSpecularBTDF->set_colour(color);
	mReflectiveBRDF->set_cr(color);
	//setSpecularColour(color);
}
Colour Transparent::shade(ShadeRec& sr)
{
	using atlas::math::Vector;

	Colour L{ Phong::shade(sr) };

	Vector wo = -sr.ray.d;
	Vector wi;
	Colour fr = mReflectiveBRDF->sample_f(sr, wo, wi);
	atlas::math::Ray<atlas::math::Vector> reflected_ray(sr.hit_point, wi);

	if (mSpecularBTDF->tir(sr)) {
		L += sr.world->tracer_ptr->trace_ray(reflected_ray, sr.depth + 1);
	}
	else {
		Vector wt;
		Colour ft = mSpecularBTDF->sample_f(sr, wo, wt);
		atlas::math::Ray<atlas::math::Vector> transmitted_ray(sr.hit_point, wt);

		L += fr * sr.world->tracer_ptr->trace_ray(reflected_ray, sr.depth + 1) *
			std::fabs(glm::dot(sr.normal, wi));

		L += fr * sr.world->tracer_ptr->trace_ray(transmitted_ray, sr.depth + 1) *
			std::fabs(glm::dot(sr.normal, wt));
	}
	return L;
}

void Transparent::setSampler(std::shared_ptr<Sampler> sampler) {
	mReflectiveBRDF->set_sampler(sampler);
	mSpecularBTDF->set_sampler(sampler);
}
void Transparent::setIor(float ior)
{
	mSpecularBTDF->set_ior(ior);
}

void Transparent::setKr(float kr)
{
	mReflectiveBRDF->set_kr(kr);
}

void Transparent::setKt(float kt)
{
	mSpecularBTDF->set_kt(kt);
}

// ******* Driver Code *******

int main()
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	std::shared_ptr<World> world{ std::make_shared<World>() };

	world->tracer_ptr = std::make_shared<Whitted>(world);

	world->width = 1200;
	world->height = 1200;
	world->max_depth = 3;
	world->background = { 0, 0, 0 };
	world->sampler = std::make_shared<MultiJittered>(64, 1440000);
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 0, -100, -200 }, 96.0f));
	world->scene[0]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour(1, 0, 0)));
	world->scene[0]->setColour({ 1, 0, 0 });

	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 128, -282, -400 }, 64.0f));
	world->scene[1]->setMaterial(
		std::make_shared<Reflective>(0.50f, 0.05f, Colour{ 0, 0, 1 }));
	world->scene[1]->setColour({ 0, 0, 1 });

	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ -128, -282, -100 }, 64.0f));
	world->scene[2]->setMaterial(
		std::make_shared<Matte>(0.1f, 0.1f, Colour{ 0, 1, 0 }));
	world->scene[2]->setColour({ 0, 1, 0 });

	world->scene.push_back(
		std::make_shared<Plane>(atlas::math::Point{ 0,0,0 }, atlas::math::Vector{ 0, -1, 0 }));
	world->scene[3]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.5,0.5,0.5 }));
	world->scene[3]->setColour({ 0.5,0.5,0.5 });


	world->scene.push_back(
		std::make_shared<Triangle>(atlas::math::Point(-223, -129, -200), atlas::math::Point(-64, -312, -300), atlas::math::Point(0, -129, -200)));

	world->scene[4]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.3,0.6,0.1 }));
	world->scene[4]->setColour({ 0.3,0.6,0.1 });

	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ -128, -282, -400 }, 64.0f));
	world->scene[5]->setMaterial(
		std::make_shared<GlossyReflector>(0.1f, 0.1f, Colour{ 1, 0, 1 }));
	world->scene[5]->setColour({ 1, 0, 1 });


	std::shared_ptr<Transparent> transparent_material{ std::make_shared<Transparent>() };
	transparent_material->setAmbientReflection(0.1f);
	transparent_material->setDiffuseReflection(0.0f);
	transparent_material->setKr(0.75f);
	transparent_material->setKt(0.9f);
	transparent_material->setIor(1.2f);
	transparent_material->setColour({ 0.95, 1, 0.95 });
	transparent_material->setSampler(std::make_shared<MultiJittered>(4, 10));

	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 356, -282, -300 }, 64.0f));
	world->scene[6]->setMaterial(transparent_material);
	world->scene[6]->setColour({ 1, 0, 1 });
	/*world->scene.push_back(
		std::make_shared<Triangle>(atlas::math::Point(-400, -64, -200), atlas::math::Point(-128, -128, -200), atlas::math::Point(-64, -64, -240)));
*/
/* world->scene[5]->setMaterial(
	 std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.5,0.1,0.6 }));
 world->scene[5]->setColour({ 0.5,0.1,0.6 });

 world->scene.push_back(
	 std::make_shared<Sphere>(atlas::math::Point{ -128, -400, -300 }, 64.0f));
 world->scene[6]->setMaterial(
	 std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.7, 1, 0.2 }));
 world->scene[6]->setColour({ 0.7, 1, 0.2 });

 world->scene.push_back(
	 std::make_shared<Plane>(atlas::math::Point{ 0,0,-3000 }, atlas::math::Vector{ 0, 0 , 1 }));
 world->scene[7]->setMaterial(
	 std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.5,0.5,0.5 }));
 world->scene[7]->setColour({ 0.5,0.5,0.5 });

 world->scene.push_back(
	 std::make_shared<Plane>(atlas::math::Point{ 0,0,-3000 }, atlas::math::Vector{ 3, 0 , 1 }));
 world->scene[8]->setMaterial(
	 std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.5,0.5,0.5 }));
 world->scene[8]->setColour({ 0.5,0.5,0.5 });

 world->scene.push_back(
	 std::make_shared<Plane>(atlas::math::Point{ 0,0,-3000 }, atlas::math::Vector{ -3, 0 , 1 }));
 world->scene[9]->setMaterial(
	 std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.6,0.6,0.6 }));
 world->scene[9]->setColour({ 0.6,0.6,0.6 });*/

 /*  world->scene.push_back(
	   std::make_shared<Box>(atlas::math::Point{300,400,-300 }, atlas::math::Point{ 500,500, -300 }));
   world->scene[10]->setMaterial(
	   std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.3,0.7,0.5 }));
   world->scene[10]->setColour({ 0.3,0.7,0.5 });*/

	world->ambient = std::make_shared<Ambient>();

	world->ambient->setColour({ 1, 1, 1 });
	world->ambient->scaleRadiance(0.05f);

	world->lights.push_back(
		std::make_shared<PointLight>(PointLight{ {0,-400,-400} }));

	/*world->lights.push_back(
		std::make_shared<Directional>(Directional{ {0, 0, 1024} }));*/

	world->lights[0]->setColour({ 1,1,1 });
	world->lights[0]->scaleRadiance(4.0f);

	/* world->lights[1]->setColour({ 1,1,1 });
	 world->lights[1]->scaleRadiance(2.0f);*/

	Pinhole camera{};

	camera.setEye({ 50.0f, -150.0f, 400.0f });
	camera.setLookAt({ 0, -25, -100 });
	camera.setDistance(500.0f);

	camera.computeUVW();

	/*camera.renderScene(world);

	saveToFile("pointlight.bmp", world->width, world->height, world->image);*/

	world->ambient = std::make_shared<AmbientOccluder>(AmbientOccluder{ 0.3f });

	world->ambient->setColour({ 1, 1, 1 });
	world->ambient->scaleRadiance(10.0f);

	//world->lights.clear();

	camera.renderScene(world);

	saveToFile("render.bmp", world->width, world->height, world->image);

	return 0;
}

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image)
{
	std::vector<unsigned char> data(image.size() * 3);

	for (std::size_t i{ 0 }, k{ 0 }; i < image.size(); ++i, k += 3)
	{
		Colour pixel = image[i];
		data[k + 0] = static_cast<unsigned char>(pixel.r * 255);
		data[k + 1] = static_cast<unsigned char>(pixel.g * 255);
		data[k + 2] = static_cast<unsigned char>(pixel.b * 255);
	}

	stbi_write_bmp(filename.c_str(),
		static_cast<int>(width),
		static_cast<int>(height),
		3,
		data.data());
}