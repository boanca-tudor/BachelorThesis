import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GenerateResultsComponent } from './generate-results.component';

describe('GenerateResultsComponent', () => {
  let component: GenerateResultsComponent;
  let fixture: ComponentFixture<GenerateResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GenerateResultsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GenerateResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
