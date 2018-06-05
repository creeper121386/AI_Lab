#include <iostream>
#include <math.h>
using namespace std;
class Point
{
  private:
    double x;
    double y;

  public:
    void set(double m, double n)
    {
        x = m;
        y = n;
    }
    void dislay()
    {
        cout << "(" << x << "," << y << ")" << endl;
    }
    double distance(Point b)
    {
        double d;
        d = sqrt((x - b.x) * (x - b.x) + (y - b.y) * (y - b.y));
        return d;
    }
};
int main()
{
    double x1, y1, x2, y2, d;
    Point a, b;
    cout << "请输入点一的横纵坐标" << endl;
    cin >> x1 >> y1;
    cout << '\n'
         << "请输入点二的横纵坐标" << endl;
    cin >> x2 >> y2;
    a.set(x1, y1);
    b.set(x2, y2);
    d = a.distance(b);
    cout << "两点间距离" << d << endl;
    return 0;
}