#ifndef IMAGE_H
#define IMAGE_H

// a simple example - you would need to add more funtions

struct RGB{
	RGB();
	RGB(int red, int green, int blue);
	RGB(const RGB& old);
	RGB& operator=(RGB);
	int r, g, b;
};

class ImageType {
 public:
   ImageType();
   ImageType(int, int, int);
   ImageType(ImageType&);
   ~ImageType();
   void getImageInfo(int&, int&, int&);
   void setImageInfo(int, int, int);
   void setPixelVal(int, int, RGB);
   void getPixelVal(int, int, RGB&);
 private:
   int N, M, Q;
   RGB **pixelValue;
};

#endif
